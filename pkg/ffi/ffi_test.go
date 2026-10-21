package ffi_test

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/pkg/ffi"
	"github.com/udaykishore-resu/gpu-boundary/pkg/memory"
)

func newQueue(t testing.TB, n int) *ffi.Queue {
	t.Helper()
	if !ffi.Available {
		t.Skip("built with CGO_ENABLED=0")
	}
	q, err := ffi.NewQueue(n)
	if err != nil {
		t.Fatalf("NewQueue: %v", err)
	}
	t.Cleanup(func() { q.Close() })
	return q
}

func TestAddCrossesBoundary(t *testing.T) {
	if !ffi.Available {
		t.Skip("built with CGO_ENABLED=0")
	}
	if got := ffi.Add(2, 40); got != 42 {
		t.Fatalf("Add(2,40) = %d, want 42", got)
	}
}

func TestQueueSaxpyMatchesReference(t *testing.T) {
	q := newQueue(t, 8)

	const n = 1000
	x := make([]float32, n)
	y := make([]float32, n)
	want := make([]float32, n)
	for i := range x {
		x[i] = float32(i) * 0.5
		y[i] = float32(n - i)
		want[i] = 2.5*x[i] + y[i]
	}

	if err := q.SubmitNow(ffi.OpSaxpy, 2.5, x, y); err != nil {
		t.Fatalf("SubmitNow: %v", err)
	}
	for i := range y {
		if math.Abs(float64(y[i]-want[i])) > 1e-4 {
			t.Fatalf("y[%d] = %v, want %v", i, y[i], want[i])
		}
	}
}

// TestBatchEqualsSequence is the correctness claim behind the whole batching
// mitigation: one crossing carrying k descriptors must produce byte-identical
// results to k crossings carrying one each.
func TestBatchEqualsSequence(t *testing.T) {
	q := newQueue(t, 16)

	const n = 257
	const k = 12

	mk := func() []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(i%17) - 8
		}
		return s
	}

	xa, ya := mk(), mk()
	xb, yb := mk(), mk()

	// One crossing per descriptor.
	for i := 0; i < k; i++ {
		if err := q.SubmitNow(ffi.OpSaxpy, float32(i)*0.25, xa, ya); err != nil {
			t.Fatalf("SubmitNow: %v", err)
		}
	}
	seqCrossings := q.Crossings

	// One crossing for all descriptors.
	for i := 0; i < k; i++ {
		if err := q.Enqueue(ffi.OpSaxpy, float32(i)*0.25, xb, yb); err != nil {
			t.Fatalf("Enqueue: %v", err)
		}
	}
	if q.Len() != k {
		t.Fatalf("Len = %d, want %d", q.Len(), k)
	}
	if err := q.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	for i := range ya {
		if ya[i] != yb[i] {
			t.Fatalf("batched result diverges at %d: %v vs %v", i, ya[i], yb[i])
		}
	}
	if got := q.Crossings - seqCrossings; got != 1 {
		t.Fatalf("batched path made %d crossings, want 1", got)
	}
	if q.Submitted != 2*k {
		t.Fatalf("Submitted = %d, want %d", q.Submitted, 2*k)
	}
	if af := q.AmplificationFactor(); af <= 1 {
		t.Fatalf("AmplificationFactor = %v, want > 1", af)
	}
}

// TestEnqueuePtrMatchesEnqueue checks that the unpinned off-heap fast path
// computes exactly what the safe pinned path computes. It is the same kernel
// either way; only the memory provenance differs.
func TestEnqueuePtrMatchesEnqueue(t *testing.T) {
	q := newQueue(t, 8)

	const n = 300
	arena, err := memory.NewArena(1 << 16)
	if err != nil {
		t.Fatal(err)
	}
	defer arena.Close()

	ax := arena.MustFloat32(n)
	ay := arena.MustFloat32(n)
	hx := make([]float32, n)
	hy := make([]float32, n)
	for i := 0; i < n; i++ {
		v := float32(i%31) - 15
		ax[i], hx[i] = v, v
		ay[i], hy[i] = -v, -v
	}

	if err := q.SubmitNow(ffi.OpSaxpy, 3.25, hx, hy); err != nil {
		t.Fatalf("Enqueue path: %v", err)
	}
	if err := q.EnqueuePtr(ffi.OpSaxpy, 3.25, memory.Pointer(ax), memory.Pointer(ay), n); err != nil {
		t.Fatalf("EnqueuePtr: %v", err)
	}
	if err := q.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	for i := 0; i < n; i++ {
		if ay[i] != hy[i] {
			t.Fatalf("off-heap result differs at %d: %v != %v", i, ay[i], hy[i])
		}
	}
}

func TestEnqueuePtrRejectsBadArgs(t *testing.T) {
	q := newQueue(t, 4)
	if err := q.EnqueuePtr(ffi.OpSaxpy, 1, nil, nil, 0); err == nil {
		t.Fatal("expected an error for a nil destination")
	}
	arena, err := memory.NewArena(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer arena.Close()
	y := arena.MustFloat32(8)
	if err := q.EnqueuePtr(ffi.OpSaxpy, 1, nil, memory.Pointer(y), 8); err == nil {
		t.Fatal("expected an error for a saxpy with no source")
	}
}

func TestQueueFull(t *testing.T) {
	q := newQueue(t, 2)
	y := make([]float32, 4)
	for i := 0; i < 2; i++ {
		if err := q.Enqueue(ffi.OpScale, 2, nil, y); err != nil {
			t.Fatalf("Enqueue %d: %v", i, err)
		}
	}
	if err := q.Enqueue(ffi.OpScale, 2, nil, y); err != ffi.ErrQueueFull {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
	if err := q.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := q.Enqueue(ffi.OpScale, 2, nil, y); err != nil {
		t.Fatalf("Enqueue after Flush: %v", err)
	}
}

func TestQueueRejectsBadArgs(t *testing.T) {
	q := newQueue(t, 4)
	if err := q.Enqueue(ffi.OpSaxpy, 1, nil, nil); err == nil {
		t.Fatal("expected error for empty destination")
	}
	if err := q.Enqueue(ffi.OpSaxpy, 1, make([]float32, 2), make([]float32, 8)); err == nil {
		t.Fatal("expected error for short source")
	}
}

func TestQueueClosedIsSafe(t *testing.T) {
	if !ffi.Available {
		t.Skip("built with CGO_ENABLED=0")
	}
	q, err := ffi.NewQueue(2)
	if err != nil {
		t.Fatalf("NewQueue: %v", err)
	}
	if err := q.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := q.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if err := q.Enqueue(ffi.OpScale, 1, nil, make([]float32, 4)); err == nil {
		t.Fatal("expected error enqueueing on a closed queue")
	}
}

// TestStallMeasurementRuns exercises the measurement itself. It asserts only
// that the harness works, not a timing threshold: CI machines are noisy and a
// test that fails on a busy runner teaches nobody anything. The numbers are
// reported by cmd/boundary-bench, where they belong.
func TestStallMeasurementRuns(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	if !ffi.Available {
		t.Skip("built with CGO_ENABLED=0")
	}
	for _, inC := range []bool{false, true} {
		res := ffi.MeasureStall(ffi.StallConfig{
			Blockers: 8,
			Hold:     2 * time.Millisecond,
			Window:   120 * time.Millisecond,
			Tick:     time.Millisecond,
			InC:      inC,
		})
		if res.Ticks == 0 {
			t.Fatalf("inC=%v: observer recorded no ticks", inC)
		}
		t.Logf("inC=%-5v threads %d->%d (+%d) late p50=%v p99=%v max=%v",
			inC, res.ThreadsBefore, res.ThreadsAfter, res.ThreadGrowth(),
			res.LateP50, res.LateP99, res.LateMax)
	}
}

func BenchmarkBoundary(b *testing.B) {
	if !ffi.Available {
		b.Skip("built with CGO_ENABLED=0")
	}
	b.Run("go-noop", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			goNoop()
		}
	})
	b.Run("cgo-noop", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ffi.Noop()
		}
	})
	b.Run("cgo-add", func(b *testing.B) {
		var s int32
		for i := 0; i < b.N; i++ {
			s = ffi.Add(s, 1)
		}
		_ = s
	})
}

//go:noinline
func goNoop() {}

// BenchmarkLaunch is the headline measurement: identical work, submitted one
// descriptor per crossing versus batch-many per crossing.
func BenchmarkLaunch(b *testing.B) {
	if !ffi.Available {
		b.Skip("built with CGO_ENABLED=0")
	}
	const elems = 64 // small kernel: boundary-dominated, like a real launch
	for _, batch := range []int{1, 4, 16, 64, 256} {
		b.Run(fmt.Sprintf("batch=%d", batch), func(b *testing.B) {
			q, err := ffi.NewQueue(batch)
			if err != nil {
				b.Fatal(err)
			}
			defer q.Close()

			x := make([]float32, elems)
			y := make([]float32, elems)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := q.Enqueue(ffi.OpSaxpy, 1.0001, x, y); err != nil {
					b.Fatal(err)
				}
				if q.Len() == batch {
					if err := q.Flush(); err != nil {
						b.Fatal(err)
					}
				}
			}
			q.Flush()
			b.StopTimer()
			b.ReportMetric(q.AmplificationFactor(), "launch/crossing")
		})
	}
}
