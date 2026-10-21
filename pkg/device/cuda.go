//go:build cuda && cgo

package device

/*
#cgo CFLAGS: -O2
#cgo linux CFLAGS: -I/usr/local/cuda/include
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcublas
#cgo windows LDFLAGS: -lcudart -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// The kernel here is cuBLAS's saxpy rather than a hand-written .cu file, and
// that is a deliberate choice: shipping a .cu means shipping nvcc as a build
// dependency, and a Go project that requires nvcc to compile has given up the
// thing that made it a Go project. Linking a vendor library is the compromise
// most Go GPU code actually makes.
//
// Everything below is plumbing that exists because cgo cannot see CUDA's
// enums, error strings or handle types in a way Go can use directly.

static cublasHandle_t gBlas = NULL;
static cudaStream_t   gStream = NULL;
static char           gErr[512];

static void set_err(const char* s) {
    if (s == NULL) s = "unknown cuda error";
    snprintf(gErr, sizeof(gErr), "%s", s);
}

const char* cbx_error(void) { return gErr; }

int cbx_init(int* nameLen, char* nameBuf, int nameCap) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 1; }
    if (count == 0)       { set_err("no CUDA devices present");  return 2; }

    struct cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 3; }
    snprintf(nameBuf, nameCap, "%s (sm_%d%d)", prop.name, prop.major, prop.minor);
    *nameLen = (int)strlen(nameBuf);

    e = cudaStreamCreate(&gStream);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 4; }

    if (cublasCreate(&gBlas) != CUBLAS_STATUS_SUCCESS) {
        set_err("cublasCreate failed"); return 5;
    }
    if (cublasSetStream(gBlas, gStream) != CUBLAS_STATUS_SUCCESS) {
        set_err("cublasSetStream failed"); return 6;
    }
    return 0;
}

void* cbx_alloc(size_t bytes) {
    void* p = NULL;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return NULL; }
    return p;
}

void cbx_free(void* p) { if (p) cudaFree(p); }

int cbx_upload(void* dst, const float* src, size_t bytes) {
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, gStream);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 1; }
    // The copy is asynchronous with respect to the host but ordered on the
    // stream, so it is safe to enqueue kernels against it immediately. It is
    // NOT safe to reuse the host buffer until the stream syncs, which is why
    // the Go side keeps the source alive across the call.
    e = cudaStreamSynchronize(gStream);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 2; }
    return 0;
}

int cbx_download(float* dst, const void* src, size_t bytes) {
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, gStream);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 1; }
    e = cudaStreamSynchronize(gStream);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 2; }
    return 0;
}

// cbx_enqueue_saxpy launches asynchronously on the stream and returns without
// waiting, so N calls cost N launches but only one synchronisation.
int cbx_enqueue_saxpy(float alpha, const void* x, void* y, int n) {
    cublasStatus_t s = cublasSaxpy(gBlas, n, &alpha, (const float*)x, 1, (float*)y, 1);
    if (s != CUBLAS_STATUS_SUCCESS) { set_err("cublasSaxpy failed"); return 1; }
    return 0;
}

int cbx_sync(void) {
    cudaError_t e = cudaStreamSynchronize(gStream);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return 1; }
    return 0;
}

void cbx_shutdown(void) {
    if (gBlas)   { cublasDestroy(gBlas);      gBlas = NULL; }
    if (gStream) { cudaStreamDestroy(gStream); gStream = NULL; }
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// cudaDevice drives an NVIDIA GPU through the CUDA runtime and cuBLAS.
//
// The interesting property, and the reason this backend exists alongside Metal,
// is that the code is structurally identical: allocate opaque handles, copy
// across an explicit boundary, enqueue asynchronously, synchronise once. Two
// completely different vendor APIs collapse to the same Go shape, which is the
// argument for putting the abstraction at this layer instead of trying to make
// Go itself the tensor language.
type cudaDevice struct {
	mu     sync.Mutex
	name   string
	stats  Stats
	closed bool
	bufs   map[*cudaBuffer]struct{}

	// pending counts launches issued on the stream since the last Sync.
	pending int
}

type cudaBuffer struct {
	dev  *cudaDevice
	ptr  unsafe.Pointer
	n    int
	free bool
}

func (b *cudaBuffer) Len() int   { return b.n }
func (b *cudaBuffer) Kind() Kind { return KindCUDA }

func (b *cudaBuffer) Free() error {
	if b == nil || b.free {
		return nil
	}
	b.dev.mu.Lock()
	defer b.dev.mu.Unlock()
	b.free = true
	C.cbx_free(b.ptr)
	b.ptr = nil
	delete(b.dev.bufs, b)
	b.dev.stats.Frees++
	return nil
}

func cudaErr(op string, code C.int) error {
	return fmt.Errorf("device: cuda %s failed (code %d): %s", op, int(code), C.GoString(C.cbx_error()))
}

func newCUDADevice() (Device, error) {
	var nameLen C.int
	buf := (*C.char)(C.malloc(256))
	defer C.free(unsafe.Pointer(buf))

	if rc := C.cbx_init(&nameLen, buf, 256); rc != 0 {
		return nil, fmt.Errorf("%w: %s", ErrUnavailable, C.GoString(C.cbx_error()))
	}
	return &cudaDevice{
		name: C.GoString(buf),
		bufs: make(map[*cudaBuffer]struct{}),
	}, nil
}

func (d *cudaDevice) Name() string { return "cuda: " + d.name }
func (d *cudaDevice) Kind() Kind   { return KindCUDA }

func (d *cudaDevice) Alloc(n int) (Buffer, error) {
	if n <= 0 {
		return nil, fmt.Errorf("device: allocation of %d elements", n)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, ErrClosed
	}
	p := C.cbx_alloc(C.size_t(n * 4))
	if p == nil {
		return nil, fmt.Errorf("device: cudaMalloc(%d) failed: %s", n*4, C.GoString(C.cbx_error()))
	}
	b := &cudaBuffer{dev: d, ptr: p, n: n}
	d.bufs[b] = struct{}{}
	d.stats.Allocs++
	return b, nil
}

func (d *cudaDevice) own(b Buffer) (*cudaBuffer, error) {
	cb, ok := b.(*cudaBuffer)
	if !ok || cb.dev != d {
		return nil, ErrForeignBuffer
	}
	if cb.free {
		return nil, errors.New("device: buffer already freed")
	}
	return cb, nil
}

func (d *cudaDevice) Upload(dst Buffer, src []float32) error {
	cb, err := d.own(dst)
	if err != nil {
		return err
	}
	if len(src) != cb.n {
		return fmt.Errorf("%w: uploading %d into %d", ErrSizeMismatch, len(src), cb.n)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return ErrClosed
	}
	rc := C.cbx_upload(cb.ptr, (*C.float)(unsafe.Pointer(&src[0])), C.size_t(len(src)*4))
	// src is Go memory holding no Go pointers, so passing it is legal, but it
	// must stay alive and unmoved until the copy has actually completed.
	runtime.KeepAlive(src)
	if rc != 0 {
		return cudaErr("upload", rc)
	}
	d.stats.Uploads++
	d.stats.BytesUp += uint64(len(src)) * 4
	return nil
}

func (d *cudaDevice) Download(dst []float32, src Buffer) error {
	cb, err := d.own(src)
	if err != nil {
		return err
	}
	if len(dst) != cb.n {
		return fmt.Errorf("%w: downloading %d into %d", ErrSizeMismatch, cb.n, len(dst))
	}
	if err := d.Sync(); err != nil {
		return err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	rc := C.cbx_download((*C.float)(unsafe.Pointer(&dst[0])), cb.ptr, C.size_t(len(dst)*4))
	runtime.KeepAlive(dst)
	if rc != 0 {
		return cudaErr("download", rc)
	}
	d.stats.Downloads++
	d.stats.BytesDown += uint64(len(dst)) * 4
	return nil
}

func (d *cudaDevice) Saxpy(alpha float32, x, y Buffer) error {
	bx, err := d.own(x)
	if err != nil {
		return err
	}
	by, err := d.own(y)
	if err != nil {
		return err
	}
	if bx.n != by.n {
		return fmt.Errorf("%w: saxpy over %d and %d", ErrSizeMismatch, bx.n, by.n)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return ErrClosed
	}
	if rc := C.cbx_enqueue_saxpy(C.float(alpha), bx.ptr, by.ptr, C.int(by.n)); rc != 0 {
		return cudaErr("saxpy", rc)
	}
	d.stats.Enqueued++
	d.pending++
	return nil
}

func (d *cudaDevice) Sync() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return ErrClosed
	}
	if d.pending == 0 {
		return nil
	}
	if rc := C.cbx_sync(); rc != 0 {
		return cudaErr("sync", rc)
	}
	d.pending = 0
	d.stats.Submissions++
	return nil
}

func (d *cudaDevice) Stats() Stats {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.stats
}

func (d *cudaDevice) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	for b := range d.bufs {
		if !b.free {
			b.free = true
			C.cbx_free(b.ptr)
			b.ptr = nil
		}
	}
	d.bufs = nil
	C.cbx_shutdown()
	return nil
}

func init() {
	Register(KindCUDA, 200, newCUDADevice)
}
