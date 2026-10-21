// Package device is a single interface over CPU, Metal and CUDA execution,
// with every accelerator backend behind a build tag.
//
// Limitation addressed: Go cannot express a GPU kernel. CUDA is a C++ dialect
// compiled by nvcc, Metal Shading Language is compiled by Apple's toolchain, and
// there is no Go equivalent of either. Whatever the backend, the kernel is
// written in another language and reached through cgo, so the honest design is
// to make that boundary explicit in the API rather than hide it.
//
// The interface is deliberately shaped like a real accelerator API rather than
// like a Go library: allocation is explicit, transfers are explicit, work is
// enqueued rather than executed, and nothing is guaranteed to have happened
// until Sync returns. A CPU backend that pretends to be synchronous would make
// the CPU path faster and the abstraction a lie.
//
// Build tags:
//
//	go build -tags metal ./...   # macOS, requires Xcode command line tools
//	go build -tags cuda  ./...   # Linux/Windows, requires CUDA toolkit + cuBLAS
//
// With neither tag the CPU backend is the only one registered and the whole
// repository still builds, tests and cross-compiles.
package device

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

// Kind identifies a backend.
type Kind string

// Known backends.
const (
	KindCPU   Kind = "cpu"
	KindMetal Kind = "metal"
	KindCUDA  Kind = "cuda"
)

// Errors returned by this package.
var (
	// ErrNotRegistered means the backend was not compiled into this binary,
	// which is usually a missing build tag rather than a missing GPU.
	ErrNotRegistered = errors.New("device: backend not registered in this build")
	// ErrUnavailable means the backend was compiled in but no usable device
	// was found at runtime.
	ErrUnavailable = errors.New("device: no usable device")
	// ErrSizeMismatch means two buffers in one operation have different
	// lengths.
	ErrSizeMismatch = errors.New("device: buffer size mismatch")
	// ErrForeignBuffer means a buffer from one device was passed to another.
	ErrForeignBuffer = errors.New("device: buffer belongs to another device")
	// ErrClosed means the device has been closed.
	ErrClosed = errors.New("device: closed")
)

// Buffer is device memory. Its contents are not readable from Go; use Download.
type Buffer interface {
	// Len is the number of float32 elements.
	Len() int
	// Kind reports which backend owns this buffer.
	Kind() Kind
	// Free releases the allocation. Freeing twice is safe.
	Free() error
}

// Device is an execution target.
//
// Saxpy enqueues work; it does not run it. Sync is the only call that promises
// anything has finished. This mirrors CUDA streams and Metal command buffers,
// and it is what makes batching measurable: N enqueues plus one Sync should
// cost dramatically less than N enqueue-and-sync pairs.
type Device interface {
	// Name is a human-readable identifier, such as the GPU model.
	Name() string
	// Kind reports the backend.
	Kind() Kind

	// Alloc reserves n float32 of device memory.
	Alloc(n int) (Buffer, error)
	// Upload copies host memory into a device buffer.
	Upload(dst Buffer, src []float32) error
	// Download copies device memory into host memory.
	Download(dst []float32, src Buffer) error

	// Saxpy enqueues y = alpha*x + y.
	Saxpy(alpha float32, x, y Buffer) error
	// Sync flushes enqueued work and blocks until it has completed.
	Sync() error

	// Stats reports counters useful for reasoning about the boundary.
	Stats() Stats
	// Close releases the device and everything still allocated on it.
	Close() error
}

// Stats counts the things worth counting at an accelerator boundary.
type Stats struct {
	// Enqueued is the number of operations submitted by the caller.
	Enqueued uint64
	// Submissions is the number of times work was actually handed to the
	// driver. Enqueued/Submissions is the batching amplification factor.
	Submissions uint64
	// Uploads and Downloads count host/device transfers.
	Uploads   uint64
	Downloads uint64
	// BytesUp and BytesDown total those transfers.
	BytesUp   uint64
	BytesDown uint64
	// Allocs and Frees track buffer lifetime, since nothing here is
	// garbage collected.
	Allocs uint64
	Frees  uint64
}

// Amplification reports operations executed per driver submission.
func (s Stats) Amplification() float64 {
	if s.Submissions == 0 {
		return 0
	}
	return float64(s.Enqueued) / float64(s.Submissions)
}

// Factory opens a device of a particular kind.
type Factory func() (Device, error)

type entry struct {
	kind Kind
	prio int
	open Factory
}

var (
	regMu sync.RWMutex
	reg   []entry
)

// Register makes a backend available to Open. Backends call this from init in
// build-tag-gated files. Higher prio wins in OpenBest.
func Register(kind Kind, prio int, f Factory) {
	regMu.Lock()
	defer regMu.Unlock()
	for i := range reg {
		if reg[i].kind == kind {
			reg[i] = entry{kind, prio, f}
			return
		}
	}
	reg = append(reg, entry{kind, prio, f})
	sort.SliceStable(reg, func(i, j int) bool { return reg[i].prio > reg[j].prio })
}

// Registered lists the backends compiled into this binary, best first.
func Registered() []Kind {
	regMu.RLock()
	defer regMu.RUnlock()
	out := make([]Kind, 0, len(reg))
	for _, e := range reg {
		out = append(out, e.kind)
	}
	return out
}

// Open opens a specific backend.
func Open(kind Kind) (Device, error) {
	regMu.RLock()
	defer regMu.RUnlock()
	for _, e := range reg {
		if e.kind == kind {
			d, err := e.open()
			if err != nil {
				return nil, fmt.Errorf("device: opening %s: %w", kind, err)
			}
			return d, nil
		}
	}
	return nil, fmt.Errorf("%w: %s (available: %v)", ErrNotRegistered, kind, Registered())
}

// OpenBest opens the highest-priority backend that actually works, falling back
// down the list. A GPU that is compiled in but absent at runtime should degrade
// to the CPU rather than fail the process.
func OpenBest() (Device, error) {
	regMu.RLock()
	entries := make([]entry, len(reg))
	copy(entries, reg)
	regMu.RUnlock()

	var errs []error
	for _, e := range entries {
		d, err := e.open()
		if err == nil {
			return d, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", e.kind, err))
	}
	if len(errs) == 0 {
		return nil, ErrNotRegistered
	}
	return nil, fmt.Errorf("device: no backend opened: %w", errors.Join(errs...))
}
