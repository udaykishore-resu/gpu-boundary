//go:build cuda && cgo

// Command cuda-launch-bench measures what FINDINGS.md section 2 measured, but
// against a real GPU instead of a C function standing in for one.
//
// The question is the same: when Go drives a device through cgo, what does each
// launch cost, how much does batching launches into one crossing recover, and
// how much of that recovery does runtime.Pinner give back when the launch
// descriptors reference Go heap memory?
//
// Each "launch" is the pattern a decode step repeats thousands of times: copy a
// small host-side parameter block to the device, then run a kernel over it.
// Concretely that is cudaMemcpyAsync of `-elems` floats followed by
// cublasSaxpy on the same stream. cuBLAS is used rather than a hand-written
// kernel for the same reason pkg/device uses it: a .cu file would make nvcc a
// build dependency.
//
// Variants, all issuing the identical CUDA work:
//
//	native C loop            one cgo call; C issues every launch. The floor:
//	                         driver + cuBLAS cost with no Go boundary at all.
//	go unbatched, heap       one cgo call per launch; host block on the Go heap
//	                         (cgo pins call arguments implicitly).
//	go unbatched, arena      one cgo call per launch; host block off-heap.
//	batch=N heap+pin         descriptors built in C memory, flushed N at a time;
//	                         every descriptor points into the Go heap, so every
//	                         one must be pinned with runtime.Pinner.
//	batch=N arena            the same, host blocks in a pkg/memory.Arena, no
//	                         pinning needed.
//	batch=N arena+registered the same arena, page-locked with cudaHostRegister,
//	                         so the copies are true asynchronous DMA.
//
// Every variant synchronises the stream every `-sync-every` launches so the
// driver's launch queue never fills; a full queue would turn a launch-cost
// measurement into a GPU-throughput measurement. After each variant the result
// buffer is read back and checked against the exact expected value, so a
// variant that silently launched nothing cannot report a flattering number.
//
// Build and run on a machine with the CUDA toolkit:
//
//	go build -tags cuda -o bin/cuda-launch-bench ./cmd/cuda-launch-bench
//	bin/cuda-launch-bench               # table
//	bin/cuda-launch-bench -json         # for cmd/bench-runs
package main

/*
#cgo CFLAGS: -O2
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcublas
#cgo windows LDFLAGS: -lcudart -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t lbBlas   = NULL;
static cudaStream_t   lbStream = NULL;
static float*         lbX      = NULL;
static float*         lbY      = NULL;
static int            lbN      = 0;
static char           lbErr[512];

static void lb_set_err(const char* where, const char* what) {
    snprintf(lbErr, sizeof(lbErr), "%s: %s", where, what ? what : "unknown error");
}

static const char* lb_error(void) { return lbErr; }

// lcmd is one launch descriptor: where the host parameter block lives, the
// scalar, and the problem size. `host` is the field that forces pinning when
// it points into the Go heap.
typedef struct {
    const float* host;
    float        alpha;
    int32_t      n;
} lcmd;

static int lb_init(int n, char* name, int nameCap, int* driverVer, int* runtimeVer) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) { lb_set_err("cudaGetDeviceCount", cudaGetErrorString(e)); return 1; }
    if (count == 0)       { lb_set_err("init", "no CUDA devices present"); return 2; }

    struct cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) { lb_set_err("cudaGetDeviceProperties", cudaGetErrorString(e)); return 3; }
    snprintf(name, nameCap, "%s (sm_%d%d)", prop.name, prop.major, prop.minor);
    cudaDriverGetVersion(driverVer);
    cudaRuntimeGetVersion(runtimeVer);

    e = cudaStreamCreateWithFlags(&lbStream, cudaStreamNonBlocking);
    if (e != cudaSuccess) { lb_set_err("cudaStreamCreate", cudaGetErrorString(e)); return 4; }
    if (cublasCreate(&lbBlas) != CUBLAS_STATUS_SUCCESS) { lb_set_err("cublasCreate", "failed"); return 5; }
    if (cublasSetStream(lbBlas, lbStream) != CUBLAS_STATUS_SUCCESS) { lb_set_err("cublasSetStream", "failed"); return 6; }

    lbN = n;
    e = cudaMalloc((void**)&lbX, (size_t)n * sizeof(float));
    if (e != cudaSuccess) { lb_set_err("cudaMalloc x", cudaGetErrorString(e)); return 7; }
    e = cudaMalloc((void**)&lbY, (size_t)n * sizeof(float));
    if (e != cudaSuccess) { lb_set_err("cudaMalloc y", cudaGetErrorString(e)); return 8; }
    return 0;
}

// lb_op is one launch: upload the host block, then saxpy over it.
static int lb_op(const float* host, float alpha, int32_t n) {
    cudaError_t e = cudaMemcpyAsync(lbX, host, (size_t)n * sizeof(float),
                                    cudaMemcpyHostToDevice, lbStream);
    if (e != cudaSuccess) { lb_set_err("cudaMemcpyAsync", cudaGetErrorString(e)); return 1; }
    if (cublasSaxpy(lbBlas, n, &alpha, lbX, 1, lbY, 1) != CUBLAS_STATUS_SUCCESS) {
        lb_set_err("cublasSaxpy", "failed"); return 2;
    }
    return 0;
}

// lb_batch issues k launches for the price of one boundary crossing.
static int lb_batch(const lcmd* cs, int32_t k) {
    int32_t i;
    for (i = 0; i < k; i++) {
        int rc = lb_op(cs[i].host, cs[i].alpha, cs[i].n);
        if (rc) return rc;
    }
    return 0;
}

static int lb_sync(void) {
    cudaError_t e = cudaStreamSynchronize(lbStream);
    if (e != cudaSuccess) { lb_set_err("cudaStreamSynchronize", cudaGetErrorString(e)); return 1; }
    return 0;
}

// lb_native is the floor: the same launches issued from C, no Go boundary.
static int lb_native(const float* host, float alpha, int32_t n, int64_t count, int32_t syncEvery) {
    int64_t i;
    for (i = 0; i < count; i++) {
        int rc = lb_op(host, alpha, n);
        if (rc) return rc;
        if (syncEvery > 0 && (i + 1) % syncEvery == 0) {
            rc = lb_sync();
            if (rc) return rc;
        }
    }
    return lb_sync();
}

static int lb_reset_y(void) {
    cudaError_t e = cudaMemsetAsync(lbY, 0, (size_t)lbN * sizeof(float), lbStream);
    if (e != cudaSuccess) { lb_set_err("cudaMemsetAsync", cudaGetErrorString(e)); return 1; }
    return lb_sync();
}

static int lb_read_y0(float* out) {
    int rc = lb_sync();
    if (rc) return rc;
    cudaError_t e = cudaMemcpy(out, lbY, sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { lb_set_err("cudaMemcpy y", cudaGetErrorString(e)); return 1; }
    return 0;
}

static int lb_register(void* p, size_t bytes) {
    cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
    if (e != cudaSuccess) { lb_set_err("cudaHostRegister", cudaGetErrorString(e)); return 1; }
    return 0;
}

static void lb_unregister(void* p) { if (p) cudaHostUnregister(p); }

static void lb_shutdown(void) {
    if (lbX)      { cudaFree(lbX); lbX = NULL; }
    if (lbY)      { cudaFree(lbY); lbY = NULL; }
    if (lbBlas)   { cublasDestroy(lbBlas); lbBlas = NULL; }
    if (lbStream) { cudaStreamDestroy(lbStream); lbStream = NULL; }
}
*/
import "C"

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"text/tabwriter"
	"time"
	"unsafe"

	"github.com/udaykishore-resu/gpu-boundary/pkg/memory"
)

// Finding and report match cmd/boundary-bench, so cmd/bench-runs can
// aggregate either.
type Finding struct {
	Section string  `json:"section"`
	Name    string  `json:"name"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
	Note    string  `json:"note"`
}

type hostInfo struct {
	GOOS        string   `json:"goos"`
	GOARCH      string   `json:"goarch"`
	NumCPU      int      `json:"num_cpu"`
	GoVersion   string   `json:"go_version"`
	CGO         bool     `json:"cgo"`
	Devices     []string `json:"devices"`
	DeviceName  string   `json:"device_name"`
	CUDADriver  string   `json:"cuda_driver"`
	CUDARuntime string   `json:"cuda_runtime"`
}

type report struct {
	Host     hostInfo  `json:"host"`
	Findings []Finding `json:"findings"`
}

const section = "cuda launch"

var (
	flagJSON      = flag.Bool("json", false, "emit JSON instead of a table")
	flagQuick     = flag.Bool("quick", false, "reduced iteration count")
	flagIters     = flag.Int("iters", 100_000, "timed launches per variant")
	flagElems     = flag.Int("elems", 16, "floats per host block; small, so the boundary dominates")
	flagSyncEvery = flag.Int("sync-every", 1024, "synchronise the stream every N launches")
	flagBatches   = flag.String("batches", "1,4,16,64,256", "batch sizes for the batched variants")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "cuda-launch-bench:", err)
		os.Exit(1)
	}
}

func cudaErr() error { return errors.New(C.GoString(C.lb_error())) }

func check(rc C.int) error {
	if rc != 0 {
		return cudaErr()
	}
	return nil
}

func run() error {
	iters := *flagIters
	if *flagQuick {
		iters = 10_000
	}
	// Every launch adds exactly 1.0 to y[0]; float32 counts exactly up to 2^24,
	// and the correctness check depends on that.
	const maxExact = 1 << 24
	warm := iters/20 + 1
	if iters+warm >= maxExact {
		return fmt.Errorf("-iters must be below %d for the exactness check", maxExact-warm)
	}
	n := *flagElems
	if n <= 0 {
		return errors.New("-elems must be positive")
	}
	batches, err := parseBatches(*flagBatches)
	if err != nil {
		return err
	}

	var nameBuf [256]C.char
	var drv, rt C.int
	if err := check(C.lb_init(C.int(n), &nameBuf[0], C.int(len(nameBuf)), &drv, &rt)); err != nil {
		return err
	}
	defer C.lb_shutdown()

	rep := report{Host: hostInfo{
		GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, NumCPU: runtime.NumCPU(),
		GoVersion: runtime.Version(), CGO: true, Devices: []string{"cuda"},
		DeviceName:  C.GoString(&nameBuf[0]),
		CUDADriver:  cudaVersion(int(drv)),
		CUDARuntime: cudaVersion(int(rt)),
	}}

	maxBatch := 1
	for _, b := range batches {
		if b > maxBatch {
			maxBatch = b
		}
	}

	// Host blocks. Every one is all ones, alpha is 1, so after k launches
	// y[0] == k exactly.
	heap := make([][]float32, maxBatch)
	for i := range heap {
		heap[i] = ones(make([]float32, n))
	}
	arenaBytes := maxBatch*(n*4+memory.DefaultAlign) + 4096
	arena, err := memory.NewArena(arenaBytes)
	if err != nil {
		return err
	}
	defer arena.Close()
	if !arena.OffHeap() {
		return errors.New("this platform has no off-heap arena; the comparison would be meaningless")
	}
	arenaBlocks := make([]unsafe.Pointer, maxBatch)
	for i := range arenaBlocks {
		arenaBlocks[i] = memory.Pointer(ones(arena.MustFloat32(n)))
	}

	m := &measurer{n: n, iters: iters, warm: warm, syncEvery: *flagSyncEvery}
	add := func(name string, v float64, note string) {
		rep.Findings = append(rep.Findings, Finding{section, name, v, "ns/launch", note})
	}

	native, err := m.time(func(k int) error {
		return check(C.lb_native((*C.float)(arenaBlocks[0]), 1, C.int32_t(n),
			C.int64_t(k), C.int32_t(m.syncEvery)))
	}, true)
	if err != nil {
		return fmt.Errorf("native: %w", err)
	}
	add("native C loop", native, "no cgo crossing per launch: the floor")

	unbHeap, err := m.time(m.unbatchedHeap(heap[0]), false)
	if err != nil {
		return fmt.Errorf("unbatched heap: %w", err)
	}
	add("go unbatched, heap", unbHeap, "one cgo call per launch")

	unbArena, err := m.time(m.unbatchedPtr(arenaBlocks[0]), false)
	if err != nil {
		return fmt.Errorf("unbatched arena: %w", err)
	}
	add("go unbatched, arena", unbArena, "one cgo call per launch")

	q, err := newQueue(maxBatch)
	if err != nil {
		return err
	}
	defer q.close()

	type row struct{ heap, arena, reg float64 }
	rows := map[int]row{}
	for _, b := range batches {
		h, err := m.time(m.batchedHeap(q, b, heap), false)
		if err != nil {
			return fmt.Errorf("batch=%d heap: %w", b, err)
		}
		a, err := m.time(m.batchedPtr(q, b, arenaBlocks), false)
		if err != nil {
			return fmt.Errorf("batch=%d arena: %w", b, err)
		}
		rows[b] = row{heap: h, arena: a}
		add(fmt.Sprintf("batch=%-3d heap+pin", b), h, fmt.Sprintf("%.2fx vs go unbatched heap", safeDiv(unbHeap, h)))
		add(fmt.Sprintf("batch=%-3d arena", b), a, fmt.Sprintf("%.2fx vs go unbatched heap", safeDiv(unbHeap, a)))
	}

	// Page-lock the arena and repeat. Registration is optional: some
	// virtualised GPUs refuse it, and the rest of the table is still valid.
	region, regionBytes := pageSpan(arenaBlocks, n*4)
	if err := check(C.lb_register(region, C.size_t(regionBytes))); err != nil {
		add("arena+registered", 0, "skipped: "+err.Error())
	} else {
		for _, b := range batches {
			r, err := m.time(m.batchedPtr(q, b, arenaBlocks), false)
			if err != nil {
				C.lb_unregister(region)
				return fmt.Errorf("batch=%d registered: %w", b, err)
			}
			row := rows[b]
			row.reg = r
			rows[b] = row
			add(fmt.Sprintf("batch=%-3d arena+registered", b), r,
				fmt.Sprintf("%.2fx vs go unbatched heap", safeDiv(unbHeap, r)))
		}
		C.lb_unregister(region)
	}

	rep.Findings = append(rep.Findings,
		Finding{section, "cgo tax per launch", unbArena - native, "ns/launch",
			"go unbatched arena minus native C loop"})
	for _, b := range []int{1, 64} {
		if r, ok := rows[b]; ok {
			rep.Findings = append(rep.Findings, Finding{section,
				fmt.Sprintf("pinning cost at batch=%d", b), r.heap - r.arena, "ns/descriptor",
				"batched heap+pin minus batched arena"})
		}
	}

	if *flagJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}
	printTable(rep)
	return nil
}

// measurer times one variant: a short untimed warmup, then iters launches,
// then a read-back that must equal warmup+iters exactly.
type measurer struct {
	n, iters, warm, syncEvery int
}

// time runs fn(k) to issue k launches. selfSyncing marks fn as already
// synchronising on its own schedule (the native loop does it inside C).
func (m *measurer) time(fn func(k int) error, selfSyncing bool) (float64, error) {
	if err := check(C.lb_reset_y()); err != nil {
		return 0, err
	}
	runtime.GC()
	if err := fn(m.warm); err != nil {
		return 0, err
	}
	if !selfSyncing {
		if err := check(C.lb_sync()); err != nil {
			return 0, err
		}
	}
	start := time.Now()
	if err := fn(m.iters); err != nil {
		return 0, err
	}
	if err := check(C.lb_sync()); err != nil {
		return 0, err
	}
	elapsed := time.Since(start)

	var y0 C.float
	if err := check(C.lb_read_y0(&y0)); err != nil {
		return 0, err
	}
	want := float32(m.warm + m.iters)
	if float32(y0) != want {
		return 0, fmt.Errorf("correctness check failed: y[0] = %v, want %v", float32(y0), want)
	}
	return float64(elapsed.Nanoseconds()) / float64(m.iters), nil
}

func (m *measurer) maybeSync(issued *int, k int) error {
	*issued += k
	if m.syncEvery > 0 && *issued >= m.syncEvery {
		*issued = 0
		return check(C.lb_sync())
	}
	return nil
}

func (m *measurer) unbatchedHeap(block []float32) func(int) error {
	return func(k int) error {
		issued := 0
		for i := 0; i < k; i++ {
			// Passing a Go pointer as a cgo argument is legal without a
			// Pinner: cgo keeps it in place for the duration of the call,
			// and a pageable cudaMemcpyAsync has staged the data before it
			// returns.
			if err := check(C.lb_op((*C.float)(unsafe.Pointer(&block[0])), 1, C.int32_t(m.n))); err != nil {
				return err
			}
			if err := m.maybeSync(&issued, 1); err != nil {
				return err
			}
		}
		return nil
	}
}

func (m *measurer) unbatchedPtr(block unsafe.Pointer) func(int) error {
	return func(k int) error {
		issued := 0
		for i := 0; i < k; i++ {
			if err := check(C.lb_op((*C.float)(block), 1, C.int32_t(m.n))); err != nil {
				return err
			}
			if err := m.maybeSync(&issued, 1); err != nil {
				return err
			}
		}
		return nil
	}
}

func (m *measurer) batchedHeap(q *queue, batch int, blocks [][]float32) func(int) error {
	return func(k int) error {
		issued := 0
		for done := 0; done < k; {
			b := min(batch, k-done)
			for i := 0; i < b; i++ {
				q.addHeap(blocks[i], m.n)
			}
			if err := q.flush(); err != nil {
				return err
			}
			done += b
			if err := m.maybeSync(&issued, b); err != nil {
				return err
			}
		}
		return nil
	}
}

func (m *measurer) batchedPtr(q *queue, batch int, blocks []unsafe.Pointer) func(int) error {
	return func(k int) error {
		issued := 0
		for done := 0; done < k; {
			b := min(batch, k-done)
			for i := 0; i < b; i++ {
				q.addPtr(blocks[i], m.n)
			}
			if err := q.flush(); err != nil {
				return err
			}
			done += b
			if err := m.maybeSync(&issued, b); err != nil {
				return err
			}
		}
		return nil
	}
}

// queue is the launch-descriptor buffer: C memory, so the descriptors
// themselves are never moved by the collector, filled from Go and submitted
// in one crossing.
type queue struct {
	cmds []C.lcmd
	buf  unsafe.Pointer
	n    int
	pin  runtime.Pinner
}

func newQueue(capacity int) (*queue, error) {
	p := C.malloc(C.size_t(capacity) * C.size_t(unsafe.Sizeof(C.lcmd{})))
	if p == nil {
		return nil, errors.New("malloc failed")
	}
	return &queue{buf: p, cmds: unsafe.Slice((*C.lcmd)(p), capacity)}, nil
}

// addHeap stores a Go heap pointer into C memory, which the cgo rules allow
// only while the pointee is pinned. This Pin is the cost being measured.
func (q *queue) addHeap(block []float32, n int) {
	q.pin.Pin(&block[0])
	c := &q.cmds[q.n]
	c.host = (*C.float)(unsafe.Pointer(&block[0]))
	c.alpha = 1
	c.n = C.int32_t(n)
	q.n++
}

func (q *queue) addPtr(block unsafe.Pointer, n int) {
	c := &q.cmds[q.n]
	c.host = (*C.float)(block)
	c.alpha = 1
	c.n = C.int32_t(n)
	q.n++
}

func (q *queue) flush() error {
	if q.n == 0 {
		return nil
	}
	err := check(C.lb_batch(&q.cmds[0], C.int32_t(q.n)))
	q.n = 0
	// Pageable copies are staged before cudaMemcpyAsync returns, so the
	// host blocks may move again as soon as the batch call is back.
	q.pin.Unpin()
	return err
}

func (q *queue) close() {
	q.pin.Unpin()
	if q.buf != nil {
		C.free(q.buf)
		q.buf = nil
		q.cmds = nil
	}
}

// pageSpan returns the page-aligned range covering every block. The arena's
// mmap base is page-aligned and the OS maps whole pages, so rounding the ends
// outward never leaves the mapping.
func pageSpan(blocks []unsafe.Pointer, blockBytes int) (unsafe.Pointer, uintptr) {
	page := uintptr(os.Getpagesize())
	lo, hi := ^uintptr(0), uintptr(0)
	for _, b := range blocks {
		p := uintptr(b)
		if p < lo {
			lo = p
		}
		if e := p + uintptr(blockBytes); e > hi {
			hi = e
		}
	}
	lo &^= page - 1
	hi = (hi + page - 1) &^ (page - 1)
	// The arithmetic stays in uintptr; converting back is valid because the
	// address is inside a live mmap region the collector never manages.
	return unsafe.Add(blocks[0], -int(uintptr(blocks[0])-lo)), hi - lo
}

func ones(s []float32) []float32 {
	for i := range s {
		s[i] = 1
	}
	return s
}

func parseBatches(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		var b int
		if _, err := fmt.Sscanf(strings.TrimSpace(f), "%d", &b); err != nil || b <= 0 {
			return nil, fmt.Errorf("bad batch size %q", f)
		}
		out = append(out, b)
	}
	if len(out) == 0 {
		return nil, errors.New("no batch sizes")
	}
	return out, nil
}

func cudaVersion(v int) string {
	if v <= 0 {
		return "unknown"
	}
	return fmt.Sprintf("%d.%d", v/1000, (v%1000)/10)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func printTable(rep report) {
	h := rep.Host
	fmt.Printf("%s — CUDA driver %s, runtime %s — %s %s/%s, %d CPUs\n\n",
		h.DeviceName, h.CUDADriver, h.CUDARuntime, h.GoVersion, h.GOOS, h.GOARCH, h.NumCPU)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "measurement\tvalue\tunit\tnote")
	for _, f := range rep.Findings {
		fmt.Fprintf(w, "%s\t%.1f\t%s\t%s\n", f.Name, f.Value, f.Unit, f.Note)
	}
	w.Flush()
}
