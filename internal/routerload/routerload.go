// Package routerload is the routing workload shared by the Go router
// benchmarks and the Python baseline in bench/python.
//
// A cross-language comparison is only worth anything if both sides do the same
// work. So the workload is fully deterministic - a fixed LCG, a fixed key
// distribution, a fixed in-flight window, a frozen clock - and both
// implementations can print a trace summary (decision counts per reason and per
// replica, plus a hash of the whole decision sequence) that must match byte for
// byte before either side's timing is reported. `make router-crosscheck` runs
// that comparison.
//
// The distribution is a hot set: 80% of requests draw from 50 prefixes, the
// rest from 950 more, which is roughly what a fleet serving a few popular
// system prompts sees. The window keeps a fixed number of requests in flight,
// releasing the oldest after each new pick, so replica load is non-zero and the
// affinity-versus-load trade-off actually gets exercised.
//
// Python must mirror every constant and every step here; bench/python/router.py
// names the Go function each piece corresponds to.
package routerload

import (
	"fmt"
	"hash/fnv"
	"strconv"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/router"
)

// Workload constants. Changing any of them changes the trace hash, which is
// the point: the Python side must be changed in the same commit.
const (
	Seed          uint64 = 0x9E3779B97F4A7C15
	Prefixes             = 1000
	HotPrefixes          = 50
	HotPercent           = 80
	Hold                 = 32
	Replicas             = 8
	MaxConcurrent        = 64
)

// lcg is Knuth's MMIX generator. It is chosen for being trivially portable
// across languages, not for statistical quality.
type lcg struct{ s uint64 }

func (g *lcg) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s
}

// Keys returns n prefix keys from the hot-set distribution.
func Keys(seed uint64, n int) []string {
	g := &lcg{s: seed}
	out := make([]string, n)
	for i := range out {
		r := (g.next() >> 33) % 100
		v := g.next() >> 33
		var idx uint64
		if r < HotPercent {
			idx = v % HotPrefixes
		} else {
			idx = HotPrefixes + v%(Prefixes-HotPrefixes)
		}
		out[i] = fmt.Sprintf("p%04d", idx)
	}
	return out
}

// epoch is the frozen clock. Bindings never expire inside a run, so the trace
// does not depend on how fast the machine is.
var epoch = time.Unix(0, 0)

// NewRouter returns a router with the workload's fleet: replicas r00, r01, ...
// each with MaxConcurrent slots and weight 1, and the router's default policy.
func NewRouter(replicas int) *router.Router {
	r := router.New(router.Config{Now: func() time.Time { return epoch }})
	for i := 0; i < replicas; i++ {
		if err := r.Add(router.Replica{ID: ReplicaID(i), MaxConcurrent: MaxConcurrent, Weight: 1}); err != nil {
			panic(err)
		}
	}
	return r
}

// ReplicaID names replica i.
func ReplicaID(i int) string { return fmt.Sprintf("r%02d", i) }

// Window keeps hold requests in flight: each Step picks a replica for a new
// request and then releases the request picked hold steps earlier.
type Window struct {
	r    *router.Router
	done []router.Done
	hits []bool
	i    int
}

// NewWindow returns a window over r.
func NewWindow(r *router.Router, hold int) *Window {
	return &Window{r: r, done: make([]router.Done, hold), hits: make([]bool, hold)}
}

// Step routes one request. A request counts as a cache hit when its decision
// was affinity, which is what a real replica would report back.
func (w *Window) Step(key string) router.Decision {
	d, done, err := w.r.Pick(key)
	if err != nil {
		panic(err) // the workload never exceeds fleet capacity
	}
	slot := w.i % len(w.done)
	if prev := w.done[slot]; prev != nil {
		prev(w.hits[slot])
	}
	w.done[slot] = done
	w.hits[slot] = d.Reason == router.ReasonAffinity
	w.i++
	return d
}

// Drain releases everything still in flight.
func (w *Window) Drain() {
	for i, d := range w.done {
		if d != nil {
			d(w.hits[i])
			w.done[i] = nil
		}
	}
}

// Summary is the cross-language trace. Field order is alphabetical by JSON
// name so that Go's encoding and Python's json.dumps(sort_keys=True) produce
// identical bytes.
type Summary struct {
	ByReason  map[string]int `json:"by_reason"`
	ByReplica map[string]int `json:"by_replica"`
	Decisions int            `json:"decisions"`
	Hash      string         `json:"hash"`
	Hold      int            `json:"hold"`
	Keys      int            `json:"keys"`
	Replicas  int            `json:"replicas"`
	Seed      uint64         `json:"seed"`
}

// Trace runs n requests through a fresh fleet and summarises every decision.
// The hash covers the exact sequence of (replica, reason) pairs, so two
// implementations that agree on the counts but differ in order still fail.
func Trace(seed uint64, n, replicas, hold int) Summary {
	r := NewRouter(replicas)
	w := NewWindow(r, hold)
	h := fnv.New64a()
	s := Summary{
		ByReason: map[string]int{}, ByReplica: map[string]int{},
		Hold: hold, Keys: n, Replicas: replicas, Seed: seed,
	}
	for _, k := range Keys(seed, n) {
		d := w.Step(k)
		s.Decisions++
		s.ByReason[string(d.Reason)]++
		s.ByReplica[d.Replica]++
		h.Write([]byte(d.Replica))
		h.Write([]byte{' '})
		h.Write([]byte(d.Reason))
		h.Write([]byte{'\n'})
	}
	w.Drain()
	s.Hash = strconv.FormatUint(h.Sum64(), 16)
	return s
}
