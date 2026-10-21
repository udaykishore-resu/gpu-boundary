// Package memory provides the allocation discipline that numeric Go code needs
// and the language does not give it.
//
// Limitations addressed:
//
//   - Go has no arenas. The arena experiment was withdrawn, so a hot path that
//     allocates per iteration hands the collector work proportional to the
//     inference rate, and the cost shows up as tail latency rather than as
//     throughput.
//   - Go gives no control over alignment, and none over page backing. A device
//     upload wants 64-byte aligned, page-aligned, ideally pinned host memory;
//     make([]float32, n) promises none of that.
//   - Go heap memory cannot be handed to C and retained. Off-heap memory can,
//     without runtime.Pinner and without the collector ever scanning it.
//
// Arena is a bump allocator over an anonymous mmap. Allocation is a pointer
// increment, freeing is a single Reset, the collector never scans the region
// because it holds no Go pointers, and the addresses are stable, so they can be
// stored in C structures directly.
//
// This is not a general purpose allocator and should not be used as one. It is
// the right shape for exactly one job: buffers whose lifetime matches a request
// or a batch.
package memory

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// Common errors.
var (
	// ErrArenaExhausted means the requested allocation did not fit. Callers
	// are expected to treat this as backpressure, not as a fatal condition:
	// a fixed arena is how you cap memory per request in the first place.
	ErrArenaExhausted = errors.New("memory: arena exhausted")
	// ErrArenaClosed means the arena's pages have been returned to the OS.
	ErrArenaClosed = errors.New("memory: arena closed")
)

// DefaultAlign is a cache-line alignment suitable for AVX-512 and for most
// device upload paths.
const DefaultAlign = 64

// Arena is a bump allocator over a fixed region of off-heap memory.
//
// An Arena is safe for concurrent use. The mutex costs about as much as a
// sync.Pool hit and buys the ability to share one region across the goroutines
// serving a single batch.
type Arena struct {
	mu     sync.Mutex
	mem    []byte
	base   uintptr
	off    uintptr
	closed bool

	// offHeap records whether mem came from mmap (true) or from the Go heap
	// (false, on platforms without mmap). Off-heap memory may be handed to C
	// and retained; heap memory may not.
	offHeap bool

	allocs    uint64
	highWater uintptr
}

// NewArena reserves size bytes. The region is rounded up to a page boundary by
// the OS; Cap reports the usable size actually requested.
func NewArena(size int) (*Arena, error) {
	if size <= 0 {
		return nil, errors.New("memory: arena size must be positive")
	}
	mem, offHeap, err := mapRegion(size)
	if err != nil {
		return nil, fmt.Errorf("memory: reserving %d bytes: %w", size, err)
	}
	a := &Arena{mem: mem, offHeap: offHeap}

	// Align the base so that offset 0 is already DefaultAlign-aligned and
	// every subsequent alignment calculation is relative to a known point.
	raw := uintptr(unsafe.Pointer(&mem[0]))
	a.base = raw
	a.off = alignUp(raw, DefaultAlign) - raw
	return a, nil
}

// OffHeap reports whether this arena's memory is invisible to the garbage
// collector and therefore safe to hand to C for an unbounded lifetime.
func (a *Arena) OffHeap() bool { return a.offHeap }

// Cap returns the total size of the region in bytes.
func (a *Arena) Cap() int { return len(a.mem) }

// Used returns the number of bytes handed out since the last Reset.
func (a *Arena) Used() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return int(a.off)
}

// HighWater returns the largest Used value observed since the arena was
// created. Sizing an arena is capacity planning, and this is the number you
// plan with.
func (a *Arena) HighWater() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return int(a.highWater)
}

// Allocs returns the number of successful allocations since creation.
func (a *Arena) Allocs() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.allocs
}

// Bytes returns n bytes aligned to align, which must be a power of two.
//
// The returned slice aliases arena memory. It is valid until Reset or Close,
// and it must not be appended to: append would silently copy it back onto the
// Go heap and defeat the entire point.
func (a *Arena) Bytes(n, align int) ([]byte, error) {
	if n < 0 {
		return nil, errors.New("memory: negative allocation")
	}
	if align <= 0 || align&(align-1) != 0 {
		return nil, fmt.Errorf("memory: align %d is not a power of two", align)
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, ErrArenaClosed
	}

	start := alignUp(a.base+a.off, uintptr(align)) - a.base
	end := start + uintptr(n)
	if end > uintptr(len(a.mem)) {
		return nil, fmt.Errorf("%w: need %d bytes at offset %d, have %d",
			ErrArenaExhausted, n, start, len(a.mem))
	}
	a.off = end
	if end > a.highWater {
		a.highWater = end
	}
	a.allocs++
	if n == 0 {
		return a.mem[start:start:start], nil
	}
	return a.mem[start:end:end], nil
}

// Float32 returns a slice of n float32 aligned to DefaultAlign.
func (a *Arena) Float32(n int) ([]float32, error) {
	b, err := a.Bytes(n*4, DefaultAlign)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n), nil
}

// MustFloat32 is Float32 for call sites where exhaustion is a programming
// error, such as tests and one-shot setup.
func (a *Arena) MustFloat32(n int) []float32 {
	s, err := a.Float32(n)
	if err != nil {
		panic(err)
	}
	return s
}

// Reset makes the whole region available again in constant time. Every slice
// previously returned by the arena is invalidated; the memory is not zeroed,
// because zeroing is the caller's decision and it is not free.
func (a *Arena) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return
	}
	a.off = alignUp(a.base, DefaultAlign) - a.base
}

// Close returns the region to the operating system. Using any slice derived
// from the arena after Close is a use-after-free: the whole point of off-heap
// memory is that the garbage collector is not there to save you.
func (a *Arena) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	a.closed = true
	mem := a.mem
	a.mem = nil
	return unmapRegion(mem, a.offHeap)
}

// Pointer returns the address of the first element of s, suitable for handing
// to C. It is only valid for slices obtained from an off-heap arena.
func Pointer(s []float32) unsafe.Pointer {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Pointer(&s[0])
}

// IsAligned reports whether s starts on an align-byte boundary.
func IsAligned(s []float32, align int) bool {
	if len(s) == 0 {
		return true
	}
	return uintptr(unsafe.Pointer(&s[0]))%uintptr(align) == 0
}

func alignUp(p, align uintptr) uintptr {
	return (p + align - 1) &^ (align - 1)
}
