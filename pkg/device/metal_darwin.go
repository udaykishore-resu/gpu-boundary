//go:build darwin && metal && cgo

package device

/*
#cgo CFLAGS: -O2
#cgo LDFLAGS: -framework Foundation -framework Metal

#include "metal_shim.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// metalDevice drives Apple's Metal compute API through the Objective-C shim.
//
// Everything interesting about this file is in what it cannot do. The kernel is
// a string of Metal Shading Language compiled at startup, because Go cannot
// express a GPU kernel. Buffers are opaque handles, because Go cannot describe
// device memory. Work is encoded into a command buffer and only submitted on
// Sync, because a submission per kernel would spend more time in the driver
// than on the GPU. None of that is a Metal quirk; the CUDA backend has the same
// shape for the same reasons.
type metalDevice struct {
	mu     sync.Mutex
	name   string
	stats  Stats
	closed bool
	bufs   map[*metalBuffer]struct{}
}

type metalBuffer struct {
	dev  *metalDevice
	ptr  unsafe.Pointer // id<MTLBuffer>
	n    int
	free bool
}

func (b *metalBuffer) Len() int   { return b.n }
func (b *metalBuffer) Kind() Kind { return KindMetal }

func (b *metalBuffer) Free() error {
	if b == nil || b.free {
		return nil
	}
	b.dev.mu.Lock()
	defer b.dev.mu.Unlock()
	b.free = true
	C.mbx_free(b.ptr)
	b.ptr = nil
	delete(b.dev.bufs, b)
	b.dev.stats.Frees++
	return nil
}

// host returns the shared-storage host view of the buffer. On Apple silicon the
// CPU and GPU address the same physical pages, so a transfer is a copy into
// mapped memory rather than a bus transfer. A discrete GPU would need a real
// DMA here and the timings would look very different.
func (b *metalBuffer) host() ([]float32, error) {
	p := C.mbx_contents(b.ptr)
	if p == nil {
		return nil, errors.New("device: metal buffer has no host mapping")
	}
	return unsafe.Slice((*float32)(p), b.n), nil
}

func metalErr(code C.int, op string) error {
	msg := C.GoString(C.mbx_error())
	if msg == "" {
		msg = "unknown metal error"
	}
	return fmt.Errorf("device: metal %s failed (code %d): %s", op, int(code), msg)
}

func newMetalDevice() (Device, error) {
	if rc := C.mbx_init(); rc != 0 {
		return nil, fmt.Errorf("%w: %s", ErrUnavailable, C.GoString(C.mbx_error()))
	}
	return &metalDevice{
		name: C.GoString(C.mbx_device_name()),
		bufs: make(map[*metalBuffer]struct{}),
	}, nil
}

func (d *metalDevice) Name() string { return "metal: " + d.name }
func (d *metalDevice) Kind() Kind   { return KindMetal }

func (d *metalDevice) Alloc(n int) (Buffer, error) {
	if n <= 0 {
		return nil, fmt.Errorf("device: allocation of %d elements", n)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, ErrClosed
	}
	p := C.mbx_alloc(C.size_t(n * 4))
	if p == nil {
		return nil, fmt.Errorf("device: metal allocation of %d bytes failed", n*4)
	}
	b := &metalBuffer{dev: d, ptr: p, n: n}
	d.bufs[b] = struct{}{}
	d.stats.Allocs++
	return b, nil
}

func (d *metalDevice) own(b Buffer) (*metalBuffer, error) {
	mb, ok := b.(*metalBuffer)
	if !ok || mb.dev != d {
		return nil, ErrForeignBuffer
	}
	if mb.free {
		return nil, errors.New("device: buffer already freed")
	}
	return mb, nil
}

func (d *metalDevice) Upload(dst Buffer, src []float32) error {
	mb, err := d.own(dst)
	if err != nil {
		return err
	}
	if len(src) != mb.n {
		return fmt.Errorf("%w: uploading %d into %d", ErrSizeMismatch, len(src), mb.n)
	}
	h, err := mb.host()
	if err != nil {
		return err
	}
	copy(h, src)
	d.mu.Lock()
	d.stats.Uploads++
	d.stats.BytesUp += uint64(len(src)) * 4
	d.mu.Unlock()
	return nil
}

func (d *metalDevice) Download(dst []float32, src Buffer) error {
	mb, err := d.own(src)
	if err != nil {
		return err
	}
	if len(dst) != mb.n {
		return fmt.Errorf("%w: downloading %d into %d", ErrSizeMismatch, mb.n, len(dst))
	}
	// Shared storage does not imply coherence with work still in flight.
	if err := d.Sync(); err != nil {
		return err
	}
	h, err := mb.host()
	if err != nil {
		return err
	}
	copy(dst, h)
	d.mu.Lock()
	d.stats.Downloads++
	d.stats.BytesDown += uint64(len(dst)) * 4
	d.mu.Unlock()
	return nil
}

func (d *metalDevice) Saxpy(alpha float32, x, y Buffer) error {
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
	if rc := C.mbx_enqueue_saxpy(C.float(alpha), bx.ptr, by.ptr, C.uint32_t(by.n)); rc != 0 {
		return metalErr(rc, "enqueue")
	}
	d.stats.Enqueued++
	return nil
}

func (d *metalDevice) Sync() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return ErrClosed
	}
	if C.mbx_pending() == 0 {
		return nil
	}
	if rc := C.mbx_commit_wait(); rc != 0 {
		return metalErr(rc, "commit")
	}
	d.stats.Submissions++
	return nil
}

func (d *metalDevice) Stats() Stats {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.stats
}

func (d *metalDevice) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	for b := range d.bufs {
		if !b.free {
			b.free = true
			C.mbx_free(b.ptr)
			b.ptr = nil
		}
	}
	d.bufs = nil
	C.mbx_shutdown()
	return nil
}

func init() {
	Register(KindMetal, 100, newMetalDevice)
}
