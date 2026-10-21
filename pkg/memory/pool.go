package memory

import (
	"math/bits"
	"sync"
	"sync/atomic"
)

// Pool recycles float32 buffers by power-of-two size class.
//
// It is the mitigation to reach for when off-heap memory is not an option: the
// buffers stay on the Go heap and stay visible to the collector, but they stop
// being garbage. A pooled buffer survives collection cycles, so the allocation
// rate in the hot path drops to roughly zero and the collector stops being
// driven by the request rate.
//
// The trade against Arena: a Pool needs no capacity plan and no Reset, but its
// memory is still Go heap memory, so it cannot be retained by C and the
// collector still scans the spans.
type Pool struct {
	// classes[k] pools buffers of capacity 1<<k.
	classes [33]sync.Pool
	once    sync.Once

	gets   atomic.Uint64
	puts   atomic.Uint64
	misses atomic.Uint64
}

// NewPool returns a ready pool. The zero Pool is also usable.
func NewPool() *Pool { return &Pool{} }

func (p *Pool) init() {
	p.once.Do(func() {
		for k := range p.classes {
			n := 1 << uint(k)
			p.classes[k].New = func() any {
				p.misses.Add(1)
				s := make([]float32, n)
				return &s
			}
		}
	})
}

// Get returns a slice of length n whose capacity is the next power of two.
// The contents are not zeroed; a caller that needs zeros should say so by
// calling GetZeroed.
func (p *Pool) Get(n int) []float32 {
	if n <= 0 {
		return nil
	}
	p.init()
	p.gets.Add(1)
	k := classOf(n)
	sp := p.classes[k].Get().(*[]float32)
	s := *sp
	return s[:n]
}

// GetZeroed is Get followed by an explicit clear, for callers that accumulate
// into the buffer rather than overwrite it.
func (p *Pool) GetZeroed(n int) []float32 {
	s := p.Get(n)
	clear(s)
	return s
}

// Put returns a buffer to the pool. Passing a slice not obtained from Get is
// harmless but pointless: it is filed by capacity, so an odd capacity simply
// lands in a smaller class and is never reused at full size.
func (p *Pool) Put(s []float32) {
	if cap(s) == 0 {
		return
	}
	p.init()
	p.puts.Add(1)
	k := classOf(cap(s))
	if 1<<uint(k) > cap(s) {
		// cap is not a power of two: file it one class down so Get never
		// hands back a slice shorter than it promised.
		k--
	}
	if k < 0 {
		return
	}
	full := s[:cap(s)][:1<<uint(k)]
	p.classes[k].Put(&full)
}

// Stats reports pool traffic. Misses counts buffers the pool had to allocate.
func (p *Pool) Stats() (gets, puts, misses uint64) {
	return p.gets.Load(), p.puts.Load(), p.misses.Load()
}

func classOf(n int) int {
	if n <= 1 {
		return 0
	}
	return bits.Len(uint(n - 1))
}
