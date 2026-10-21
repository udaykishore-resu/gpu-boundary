package memory_test

import (
	"errors"
	"testing"

	"github.com/udaykishore-resu/gpu-boundary/pkg/memory"
)

func TestArenaAllocateAlignReset(t *testing.T) {
	a, err := memory.NewArena(1 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	t.Logf("arena is off-heap: %v", a.OffHeap())

	var first []float32
	for i := 0; i < 64; i++ {
		// Deliberately awkward sizes so alignment has to do real work.
		s, err := a.Float32(3 + i)
		if err != nil {
			t.Fatalf("Float32(%d): %v", 3+i, err)
		}
		if len(s) != 3+i {
			t.Fatalf("len = %d, want %d", len(s), 3+i)
		}
		if !memory.IsAligned(s, memory.DefaultAlign) {
			t.Fatalf("allocation %d is not %d-byte aligned", i, memory.DefaultAlign)
		}
		// Write through the slice to prove the pages are mapped writable.
		for j := range s {
			s[j] = float32(j)
		}
		if i == 0 {
			first = s
		}
	}

	if a.Used() == 0 {
		t.Fatal("Used is zero after 64 allocations")
	}
	if a.Allocs() != 64 {
		t.Fatalf("Allocs = %d, want 64", a.Allocs())
	}
	high := a.HighWater()

	a.Reset()
	if a.Used() != 0 {
		t.Fatalf("Used = %d after Reset, want 0", a.Used())
	}
	if a.HighWater() != high {
		t.Fatalf("HighWater changed across Reset: %d != %d", a.HighWater(), high)
	}

	// After Reset the first allocation must land back at the start.
	again, err := a.Float32(3)
	if err != nil {
		t.Fatalf("Float32 after Reset: %v", err)
	}
	if memory.Pointer(again) != memory.Pointer(first) {
		t.Fatal("Reset did not rewind the bump pointer")
	}
}

func TestArenaExhaustion(t *testing.T) {
	a, err := memory.NewArena(4096)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	if _, err := a.Float32(1 << 20); !errors.Is(err, memory.ErrArenaExhausted) {
		t.Fatalf("expected ErrArenaExhausted, got %v", err)
	}
	// Exhaustion must not corrupt the arena.
	if _, err := a.Float32(8); err != nil {
		t.Fatalf("arena unusable after a failed allocation: %v", err)
	}
}

func TestArenaClosed(t *testing.T) {
	a, err := memory.NewArena(4096)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := a.Float32(1); !errors.Is(err, memory.ErrArenaClosed) {
		t.Fatalf("expected ErrArenaClosed, got %v", err)
	}
	a.Reset() // must not panic
}

func TestArenaRejectsBadAlign(t *testing.T) {
	a, _ := memory.NewArena(4096)
	defer a.Close()
	if _, err := a.Bytes(16, 3); err == nil {
		t.Fatal("expected error for non-power-of-two alignment")
	}
}

func TestArenaConcurrent(t *testing.T) {
	a, err := memory.NewArena(8 << 20)
	if err != nil {
		t.Fatalf("NewArena: %v", err)
	}
	defer a.Close()

	const workers = 8
	const each = 200
	done := make(chan struct{}, workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for i := 0; i < each; i++ {
				s, err := a.Float32(64)
				if err != nil {
					t.Errorf("Float32: %v", err)
					return
				}
				for j := range s {
					s[j] = 1
				}
			}
		}()
	}
	for w := 0; w < workers; w++ {
		<-done
	}
	if got := a.Allocs(); got != workers*each {
		t.Fatalf("Allocs = %d, want %d", got, workers*each)
	}
}

func TestPoolRoundTrip(t *testing.T) {
	p := memory.NewPool()
	for _, n := range []int{1, 2, 3, 7, 8, 100, 1000, 4096, 4097} {
		s := p.Get(n)
		if len(s) != n {
			t.Fatalf("Get(%d) returned len %d", n, len(s))
		}
		if cap(s) < n {
			t.Fatalf("Get(%d) returned cap %d", n, cap(s))
		}
		for i := range s {
			s[i] = float32(i)
		}
		p.Put(s)
	}
	gets, puts, misses := p.Stats()
	if gets != puts {
		t.Fatalf("gets %d != puts %d", gets, puts)
	}
	t.Logf("pool: gets=%d puts=%d misses=%d", gets, puts, misses)
}

func TestPoolReuses(t *testing.T) {
	p := memory.NewPool()
	// Warm one size class, then hammer it.
	p.Put(p.Get(1024))
	_, _, before := p.Stats()
	const n = 1000
	for i := 0; i < n; i++ {
		p.Put(p.Get(1024))
	}
	after := func() uint64 { _, _, m := p.Stats(); return m }()

	misses := after - before
	t.Logf("pool missed %d of %d reuses (%.0f%% hit rate)", misses, n,
		100*(1-float64(misses)/float64(n)))

	// The bound is deliberately loose. Without the race detector this loop
	// misses zero times; with it enabled the runtime collects far more often
	// and sync.Pool is drained on every cycle, so the same loop misses around
	// 25%. That is an artefact of the instrumentation, not of the pool, and a
	// tight bound here would just make CI flaky. What is being asserted is
	// that reuse happens at all.
	if misses > n/2 {
		t.Fatalf("pool missed %d of %d reuses of a single size class: it is not reusing", misses, n)
	}
}

func TestPoolGetZeroed(t *testing.T) {
	p := memory.NewPool()
	s := p.Get(64)
	for i := range s {
		s[i] = 7
	}
	p.Put(s)
	z := p.GetZeroed(64)
	for i, v := range z {
		if v != 0 {
			t.Fatalf("GetZeroed[%d] = %v, want 0", i, v)
		}
	}
}

func BenchmarkAllocation(b *testing.B) {
	const n = 4096

	b.Run("make", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			s := make([]float32, n)
			s[0] = 1
			runtimeKeepAlive(s)
		}
	})

	b.Run("pool", func(b *testing.B) {
		p := memory.NewPool()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := p.Get(n)
			s[0] = 1
			p.Put(s)
		}
	})

	b.Run("arena", func(b *testing.B) {
		a, err := memory.NewArena(1 << 22)
		if err != nil {
			b.Fatal(err)
		}
		defer a.Close()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s, err := a.Float32(n)
			if err != nil {
				a.Reset()
				s, err = a.Float32(n)
				if err != nil {
					b.Fatal(err)
				}
			}
			s[0] = 1
		}
	})
}

//go:noinline
func runtimeKeepAlive(s []float32) {}
