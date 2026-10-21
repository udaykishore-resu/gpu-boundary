package gcprobe_test

import (
	"math"
	"testing"

	"github.com/udaykishore-resu/gpu-boundary/pkg/gcprobe"
)

// TestStrategiesComputeTheSameThing is the guard that makes the comparison
// meaningful: if pooling or arena allocation changed the arithmetic, the
// latency numbers would not be comparable.
func TestStrategiesComputeTheSameThing(t *testing.T) {
	cfg := gcprobe.Config{Elems: 4096, Iterations: 200, Warmup: 20}
	results, err := gcprobe.RunAll(cfg)
	if err != nil {
		t.Fatalf("RunAll: %v", err)
	}
	if len(results) != len(gcprobe.Strategies) {
		t.Fatalf("got %d results, want %d", len(results), len(gcprobe.Strategies))
	}
	want := results[0].Checksum
	for _, r := range results {
		if math.Abs(r.Checksum-want) > 1e-3 {
			t.Errorf("strategy %s checksum %v, want %v", r.Config.Strategy, r.Checksum, want)
		}
		if r.P50 <= 0 {
			t.Errorf("strategy %s recorded a non-positive p50", r.Config.Strategy)
		}
		t.Log(r)
	}
}

// TestArenaAllocatesNothing asserts the property the arena exists for. It does
// not assert a latency improvement: on a two-core CI runner the collector is
// cheap enough that the tails overlap, and a flaky performance assertion is
// worse than no assertion. Allocation volume is deterministic, so that is what
// gets asserted.
func TestArenaAllocatesNothing(t *testing.T) {
	cfg := gcprobe.Config{Elems: 8192, Iterations: 500, Warmup: 50}

	heapCfg, arenaCfg := cfg, cfg
	heapCfg.Strategy = gcprobe.StrategyHeap
	arenaCfg.Strategy = gcprobe.StrategyArena

	heap, err := gcprobe.Run(heapCfg)
	if err != nil {
		t.Fatalf("heap run: %v", err)
	}
	arena, err := gcprobe.Run(arenaCfg)
	if err != nil {
		t.Fatalf("arena run: %v", err)
	}

	t.Logf("heap : %s (%.1f MB allocated, %d mallocs)", heap, heap.TotalAllocMB, heap.Mallocs)
	t.Logf("arena: %s (%.1f MB allocated, %d mallocs)", arena, arena.TotalAllocMB, arena.Mallocs)

	if arena.TotalAllocMB >= heap.TotalAllocMB {
		t.Errorf("arena allocated %.2f MB, heap allocated %.2f MB: arena should allocate far less",
			arena.TotalAllocMB, heap.TotalAllocMB)
	}
	// The heap strategy allocates two 32 KiB buffers per iteration, so it
	// should be moving tens of megabytes; the arena should be near zero.
	if arena.TotalAllocMB > 1.0 {
		t.Errorf("arena allocated %.2f MB, expected under 1 MB", arena.TotalAllocMB)
	}
}

func TestUnknownStrategy(t *testing.T) {
	if _, err := gcprobe.Run(gcprobe.Config{Strategy: "nonsense", Iterations: 1}); err == nil {
		t.Fatal("expected an error for an unknown strategy")
	}
}

func BenchmarkStrategies(b *testing.B) {
	for _, s := range gcprobe.Strategies {
		b.Run(string(s), func(b *testing.B) {
			_, err := gcprobe.Run(gcprobe.Config{
				Strategy:   s,
				Elems:      8192,
				Iterations: b.N,
				Warmup:     1,
			})
			if err != nil {
				b.Fatal(err)
			}
		})
	}
}
