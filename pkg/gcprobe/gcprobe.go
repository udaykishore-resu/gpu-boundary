// Package gcprobe quantifies what the garbage collector costs an inference hot
// path, and what the mitigations in pkg/memory buy back.
//
// Limitation addressed: Go's collector is concurrent and low-pause, but it is
// not free and it is not absent. A serving loop that allocates a tensor per
// request drives the collector at the request rate. Throughput barely moves,
// which is why this looks fine in a benchmark, and the tail moves a lot, which
// is why it does not look fine in production. Inference SLOs are written
// against p99, so the tail is the number that matters.
//
// The probe runs the same arithmetic three ways - allocating per iteration,
// recycling through a sync.Pool, and bump-allocating from an off-heap arena -
// and reports the per-iteration latency distribution alongside collector
// statistics for each.
package gcprobe

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sort"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/pkg/kernel"
	"github.com/udaykishore-resu/gpu-boundary/pkg/memory"
)

// Strategy selects how the workload obtains its working buffers.
type Strategy string

// Available allocation strategies.
const (
	// StrategyHeap allocates a fresh buffer per iteration. This is what
	// idiomatic Go looks like and what most services actually do.
	StrategyHeap Strategy = "heap"
	// StrategyPool recycles buffers through a size-classed sync.Pool.
	StrategyPool Strategy = "pool"
	// StrategyArena bump-allocates from an off-heap region reset per
	// iteration.
	StrategyArena Strategy = "arena"
)

// Strategies lists every strategy in the order the report should present them.
var Strategies = []Strategy{StrategyHeap, StrategyPool, StrategyArena}

// Config describes one probe run.
type Config struct {
	// Strategy selects the allocation approach.
	Strategy Strategy
	// Elems is the size of the working buffers, in float32.
	Elems int
	// Iterations is how many times the workload runs.
	Iterations int
	// GOGC overrides the collector target for the duration of the run. Zero
	// leaves it alone. Setting it low is a fair way to simulate a service
	// running near its memory limit.
	GOGC int
	// Warmup iterations are executed and discarded before measurement.
	Warmup int
}

func (c Config) withDefaults() Config {
	if c.Elems <= 0 {
		c.Elems = 8192
	}
	if c.Iterations <= 0 {
		c.Iterations = 20000
	}
	if c.Warmup <= 0 {
		c.Warmup = c.Iterations / 20
	}
	return c
}

// Result is what one probe run measured.
type Result struct {
	Config Config

	Wall       time.Duration
	NsPerIter  float64
	Throughput float64 // iterations per second

	// Per-iteration latency distribution. The tail is the finding.
	P50  time.Duration
	P99  time.Duration
	P999 time.Duration
	Max  time.Duration

	// Collector statistics observed across the measured window.
	NumGC          uint32
	PauseTotal     time.Duration
	PauseMax       time.Duration
	HeapAllocDelta int64
	TotalAllocMB   float64
	Mallocs        uint64

	// Checksum keeps the compiler from eliminating the workload and lets the
	// caller confirm the strategies computed the same thing.
	Checksum float64
}

// String renders one line suitable for a table.
func (r Result) String() string {
	return fmt.Sprintf("%-6s p50=%-10v p99=%-10v p999=%-10v max=%-10v gc=%-4d pause_max=%-10v alloc=%.1fMB",
		r.Config.Strategy, r.P50, r.P99, r.P999, r.Max, r.NumGC, r.PauseMax, r.TotalAllocMB)
}

// Run executes one probe.
//
// The workload is deliberately a real kernel rather than a spin loop: it fills
// two buffers and takes their inner product using the best implementation the
// host offers, so the allocation cost is measured against arithmetic that a
// serving path would actually be doing.
func Run(cfg Config) (Result, error) {
	cfg = cfg.withDefaults()
	res := Result{Config: cfg}

	if cfg.GOGC != 0 {
		prev := debug.SetGCPercent(cfg.GOGC)
		defer debug.SetGCPercent(prev)
	}

	alloc, release, cleanup, err := allocator(cfg)
	if err != nil {
		return res, err
	}
	defer cleanup()

	work := func() float64 {
		a, b := alloc()
		for i := range a {
			f := float32(i & 1023)
			a[i] = f
			b[i] = 1 - f
		}
		v := float64(kernel.Dot(a, b))
		release(a, b)
		return v
	}

	for i := 0; i < cfg.Warmup; i++ {
		res.Checksum = work()
	}

	// Start from a clean slate so the measured window attributes collector
	// work to this run and not to the warmup.
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)

	lat := make([]time.Duration, cfg.Iterations)
	start := time.Now()
	for i := 0; i < cfg.Iterations; i++ {
		t0 := time.Now()
		res.Checksum = work()
		lat[i] = time.Since(t0)
	}
	res.Wall = time.Since(start)
	runtime.ReadMemStats(&after)

	res.NsPerIter = float64(res.Wall.Nanoseconds()) / float64(cfg.Iterations)
	if res.Wall > 0 {
		res.Throughput = float64(cfg.Iterations) / res.Wall.Seconds()
	}

	res.NumGC = after.NumGC - before.NumGC
	res.PauseTotal = time.Duration(after.PauseTotalNs - before.PauseTotalNs)
	res.PauseMax = maxPause(&before, &after)
	res.HeapAllocDelta = int64(after.HeapAlloc) - int64(before.HeapAlloc)
	res.TotalAllocMB = float64(after.TotalAlloc-before.TotalAlloc) / (1 << 20)
	res.Mallocs = after.Mallocs - before.Mallocs

	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	res.P50 = pct(lat, 0.50)
	res.P99 = pct(lat, 0.99)
	res.P999 = pct(lat, 0.999)
	res.Max = lat[len(lat)-1]
	return res, nil
}

// RunAll executes every strategy with the same shape of workload, which is the
// only way the numbers are comparable.
func RunAll(base Config) ([]Result, error) {
	out := make([]Result, 0, len(Strategies))
	for _, s := range Strategies {
		c := base
		c.Strategy = s
		r, err := Run(c)
		if err != nil {
			return out, fmt.Errorf("gcprobe: strategy %s: %w", s, err)
		}
		out = append(out, r)
	}
	return out, nil
}

func allocator(cfg Config) (alloc func() ([]float32, []float32), release func(a, b []float32), cleanup func(), err error) {
	n := cfg.Elems
	switch cfg.Strategy {
	case StrategyHeap:
		return func() ([]float32, []float32) {
				return make([]float32, n), make([]float32, n)
			},
			func(a, b []float32) {},
			func() {},
			nil

	case StrategyPool:
		p := memory.NewPool()
		return func() ([]float32, []float32) {
				return p.Get(n), p.Get(n)
			},
			func(a, b []float32) {
				p.Put(a)
				p.Put(b)
			},
			func() {},
			nil

	case StrategyArena:
		// Two buffers per iteration plus alignment slack.
		a, aerr := memory.NewArena(2*n*4 + 4*memory.DefaultAlign)
		if aerr != nil {
			return nil, nil, nil, aerr
		}
		return func() ([]float32, []float32) {
				a.Reset()
				return a.MustFloat32(n), a.MustFloat32(n)
			},
			func(x, y []float32) {},
			func() { a.Close() },
			nil

	default:
		return nil, nil, nil, fmt.Errorf("gcprobe: unknown strategy %q", cfg.Strategy)
	}
}

func pct(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)-1))
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

// maxPause walks the runtime's circular pause buffer for the pauses that
// happened between the two samples.
func maxPause(before, after *runtime.MemStats) time.Duration {
	n := after.NumGC - before.NumGC
	if n == 0 {
		return 0
	}
	if n > uint32(len(after.PauseNs)) {
		n = uint32(len(after.PauseNs))
	}
	var m uint64
	for i := uint32(0); i < n; i++ {
		idx := (after.NumGC - 1 - i) % uint32(len(after.PauseNs))
		if p := after.PauseNs[idx]; p > m {
			m = p
		}
	}
	return time.Duration(m)
}
