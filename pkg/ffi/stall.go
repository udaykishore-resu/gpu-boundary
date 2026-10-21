//go:build cgo

package ffi

import (
	"runtime"
	"runtime/pprof"
	"sort"
	"sync"
	"time"
)

// StallConfig describes a scheduler stall measurement.
type StallConfig struct {
	// Blockers is the number of goroutines held in a blocking call.
	Blockers int
	// Hold is how long each blocking call lasts.
	Hold time.Duration
	// Window is the total measurement duration.
	Window time.Duration
	// Tick is the interval the observer goroutine tries to wake on.
	Tick time.Duration
	// InC selects where the blockers block: inside C (true) or inside Go
	// (false). The Go case is the control.
	InC bool
}

func (c StallConfig) withDefaults() StallConfig {
	if c.Blockers <= 0 {
		c.Blockers = runtime.GOMAXPROCS(0) * 4
	}
	if c.Hold <= 0 {
		c.Hold = 2 * time.Millisecond
	}
	if c.Window <= 0 {
		c.Window = 300 * time.Millisecond
	}
	if c.Tick <= 0 {
		c.Tick = time.Millisecond
	}
	return c
}

// StallResult reports what the observer goroutine experienced.
type StallResult struct {
	Config StallConfig

	// ThreadsBefore and ThreadsAfter come from the threadcreate profile. The
	// growth is the runtime spawning OS threads to replace the ones parked
	// inside C.
	ThreadsBefore int
	ThreadsAfter  int

	// Ticks observed, and how late each one was relative to Tick.
	Ticks     int
	LateP50   time.Duration
	LateP99   time.Duration
	LateMax   time.Duration
	Completed int
}

// ThreadGrowth reports how many OS threads the runtime created during the run.
func (r StallResult) ThreadGrowth() int { return r.ThreadsAfter - r.ThreadsBefore }

// MeasureStall runs Blockers goroutines that each repeatedly block for Hold,
// either inside C or inside Go, while an observer goroutine measures how late
// its own wakeups are.
//
// The finding this produces: blocking in Go costs the observer almost nothing,
// because the runtime parks the goroutine and reuses the thread. Blocking in C
// forces the runtime to hand off the P and create a thread, so thread count
// climbs with concurrency and wakeup latency degrades. Any Go program that
// calls a synchronous GPU API from many goroutines pays this.
func MeasureStall(cfg StallConfig) StallResult {
	cfg = cfg.withDefaults()
	res := StallResult{Config: cfg}
	res.ThreadsBefore = threadCount()

	stop := make(chan struct{})
	var wg sync.WaitGroup
	var completed struct {
		sync.Mutex
		n int
	}

	for i := 0; i < cfg.Blockers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			local := 0
			for {
				select {
				case <-stop:
					completed.Lock()
					completed.n += local
					completed.Unlock()
					return
				default:
				}
				if cfg.InC {
					Block(cfg.Hold)
				} else {
					time.Sleep(cfg.Hold)
				}
				local++
			}
		}()
	}

	// Give the blockers a moment to actually be blocking before sampling.
	time.Sleep(10 * time.Millisecond)

	late := make([]time.Duration, 0, int(cfg.Window/cfg.Tick)+8)
	deadline := time.Now().Add(cfg.Window)
	for time.Now().Before(deadline) {
		start := time.Now()
		time.Sleep(cfg.Tick)
		d := time.Since(start) - cfg.Tick
		if d < 0 {
			d = 0
		}
		late = append(late, d)
	}

	res.ThreadsAfter = threadCount()
	close(stop)
	wg.Wait()

	completed.Lock()
	res.Completed = completed.n
	completed.Unlock()

	res.Ticks = len(late)
	sort.Slice(late, func(i, j int) bool { return late[i] < late[j] })
	res.LateP50 = quantile(late, 0.50)
	res.LateP99 = quantile(late, 0.99)
	if len(late) > 0 {
		res.LateMax = late[len(late)-1]
	}
	return res
}

func quantile(sorted []time.Duration, q float64) time.Duration {
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

func threadCount() int {
	p := pprof.Lookup("threadcreate")
	if p == nil {
		return 0
	}
	return p.Count()
}
