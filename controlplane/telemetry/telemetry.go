// Package telemetry is a dependency-free Prometheus exposition endpoint.
//
// The obvious move is to import the official client. This repository does not,
// for one reason worth stating: the whole argument is that Go's advantage in
// this domain is operational, and an operational argument is weaker if it
// depends on the reader trusting a dependency tree. Counters, gauges and
// histograms in the text exposition format are about two hundred lines, they
// have no moving parts, and writing them here keeps the module at zero
// dependencies and therefore trivially auditable.
//
// For production, use the official client. For a reference implementation whose
// entire supply chain is the standard library, this is the honest version.
package telemetry

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// Registry holds metrics and renders them.
type Registry struct {
	mu      sync.RWMutex
	metrics []metric
	names   map[string]struct{}
}

type metric interface {
	name() string
	help() string
	kind() string
	write(w io.Writer)
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{names: make(map[string]struct{})}
}

func (r *Registry) register(m metric) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, dup := r.names[m.name()]; dup {
		panic("telemetry: duplicate metric name " + m.name())
	}
	r.names[m.name()] = struct{}{}
	r.metrics = append(r.metrics, m)
}

// WriteTo renders the registry in Prometheus text exposition format.
func (r *Registry) WriteTo(w io.Writer) (int64, error) {
	r.mu.RLock()
	ms := make([]metric, len(r.metrics))
	copy(ms, r.metrics)
	r.mu.RUnlock()

	sort.Slice(ms, func(i, j int) bool { return ms[i].name() < ms[j].name() })

	var buf strings.Builder
	for _, m := range ms {
		fmt.Fprintf(&buf, "# HELP %s %s\n", m.name(), m.help())
		fmt.Fprintf(&buf, "# TYPE %s %s\n", m.name(), m.kind())
		m.write(&buf)
	}
	n, err := io.WriteString(w, buf.String())
	return int64(n), err
}

// Handler serves the registry over HTTP.
func (r *Registry) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		if _, err := r.WriteTo(w); err != nil {
			// The response is already partly written, so there is nothing
			// useful to say to the client; the scrape will fail and be
			// retried, which is the correct outcome.
			return
		}
	})
}

// labelSet is a canonical rendering of label values.
func labelSet(names, values []string) string {
	if len(names) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteByte('{')
	for i, n := range names {
		if i > 0 {
			b.WriteByte(',')
		}
		v := ""
		if i < len(values) {
			v = values[i]
		}
		b.WriteString(n)
		b.WriteString(`="`)
		b.WriteString(escape(v))
		b.WriteString(`"`)
	}
	b.WriteByte('}')
	return b.String()
}

func escape(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return r.Replace(s)
}

// ---------------------------------------------------------------- counters

// Counter is a monotonically increasing value, optionally with labels.
type Counter struct {
	n, h   string
	labels []string

	mu   sync.RWMutex
	vals map[string]*atomic.Uint64
	keys []string
}

// NewCounter registers a counter.
func NewCounter(r *Registry, name, help string, labels ...string) *Counter {
	c := &Counter{n: name, h: help, labels: labels, vals: make(map[string]*atomic.Uint64)}
	r.register(c)
	return c
}

func (c *Counter) name() string { return c.n }
func (c *Counter) help() string { return c.h }
func (c *Counter) kind() string { return "counter" }

func (c *Counter) slot(values []string) *atomic.Uint64 {
	key := labelSet(c.labels, values)
	c.mu.RLock()
	v, ok := c.vals[key]
	c.mu.RUnlock()
	if ok {
		return v
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.vals[key]; ok {
		return v
	}
	v = new(atomic.Uint64)
	c.vals[key] = v
	c.keys = append(c.keys, key)
	sort.Strings(c.keys)
	return v
}

// Inc adds one.
func (c *Counter) Inc(values ...string) { c.slot(values).Add(1) }

// Add adds n.
func (c *Counter) Add(n uint64, values ...string) { c.slot(values).Add(n) }

// Value reads the current count for a label combination.
func (c *Counter) Value(values ...string) uint64 { return c.slot(values).Load() }

func (c *Counter) write(w io.Writer) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if len(c.keys) == 0 {
		fmt.Fprintf(w, "%s 0\n", c.n)
		return
	}
	for _, k := range c.keys {
		fmt.Fprintf(w, "%s%s %d\n", c.n, k, c.vals[k].Load())
	}
}

// ------------------------------------------------------------------ gauges

// Gauge is a value that goes up and down.
type Gauge struct {
	n, h   string
	labels []string

	mu   sync.RWMutex
	vals map[string]*atomic.Uint64 // float64 bits
	keys []string
}

// NewGauge registers a gauge.
func NewGauge(r *Registry, name, help string, labels ...string) *Gauge {
	g := &Gauge{n: name, h: help, labels: labels, vals: make(map[string]*atomic.Uint64)}
	r.register(g)
	return g
}

func (g *Gauge) name() string { return g.n }
func (g *Gauge) help() string { return g.h }
func (g *Gauge) kind() string { return "gauge" }

func (g *Gauge) slot(values []string) *atomic.Uint64 {
	key := labelSet(g.labels, values)
	g.mu.RLock()
	v, ok := g.vals[key]
	g.mu.RUnlock()
	if ok {
		return v
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if v, ok := g.vals[key]; ok {
		return v
	}
	v = new(atomic.Uint64)
	g.vals[key] = v
	g.keys = append(g.keys, key)
	sort.Strings(g.keys)
	return v
}

// Set stores v.
func (g *Gauge) Set(v float64, values ...string) {
	g.slot(values).Store(math.Float64bits(v))
}

// Value reads the gauge.
func (g *Gauge) Value(values ...string) float64 {
	return math.Float64frombits(g.slot(values).Load())
}

func (g *Gauge) write(w io.Writer) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if len(g.keys) == 0 {
		fmt.Fprintf(w, "%s 0\n", g.n)
		return
	}
	for _, k := range g.keys {
		fmt.Fprintf(w, "%s%s %s\n", g.n, k, formatFloat(math.Float64frombits(g.vals[k].Load())))
	}
}

// -------------------------------------------------------------- histograms

// DefaultBuckets are latency buckets in seconds, spanning the range an
// inference gateway actually cares about: sub-millisecond routing decisions
// through multi-second generations.
var DefaultBuckets = []float64{
	0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10,
}

// Histogram counts observations into cumulative buckets.
type Histogram struct {
	n, h    string
	labels  []string
	buckets []float64

	mu   sync.RWMutex
	vals map[string]*histVal
	keys []string
}

type histVal struct {
	counts []atomic.Uint64
	sum    atomic.Uint64 // float64 bits, updated under CAS
	total  atomic.Uint64
}

// NewHistogram registers a histogram. Passing nil buckets uses DefaultBuckets.
func NewHistogram(r *Registry, name, help string, buckets []float64, labels ...string) *Histogram {
	if buckets == nil {
		buckets = DefaultBuckets
	}
	b := make([]float64, len(buckets))
	copy(b, buckets)
	sort.Float64s(b)
	hst := &Histogram{n: name, h: help, labels: labels, buckets: b, vals: make(map[string]*histVal)}
	r.register(hst)
	return hst
}

func (h *Histogram) name() string { return h.n }
func (h *Histogram) help() string { return h.h }
func (h *Histogram) kind() string { return "histogram" }

func (h *Histogram) slot(values []string) *histVal {
	key := labelSet(h.labels, values)
	h.mu.RLock()
	v, ok := h.vals[key]
	h.mu.RUnlock()
	if ok {
		return v
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if v, ok := h.vals[key]; ok {
		return v
	}
	v = &histVal{counts: make([]atomic.Uint64, len(h.buckets)+1)}
	h.vals[key] = v
	h.keys = append(h.keys, key)
	sort.Strings(h.keys)
	return v
}

// Observe records a value, in the histogram's unit (seconds, by convention).
func (h *Histogram) Observe(v float64, values ...string) {
	hv := h.slot(values)
	idx := sort.SearchFloat64s(h.buckets, v)
	// SearchFloat64s finds the first bucket >= v; Prometheus buckets are
	// "less than or equal", so an exact boundary hit belongs in that bucket.
	if idx < len(h.buckets) && h.buckets[idx] < v {
		idx++
	}
	hv.counts[idx].Add(1)
	hv.total.Add(1)
	for {
		old := hv.sum.Load()
		nv := math.Float64bits(math.Float64frombits(old) + v)
		if hv.sum.CompareAndSwap(old, nv) {
			break
		}
	}
}

// Count reports how many observations were recorded.
func (h *Histogram) Count(values ...string) uint64 { return h.slot(values).total.Load() }

// Sum reports the sum of observations.
func (h *Histogram) Sum(values ...string) float64 {
	return math.Float64frombits(h.slot(values).sum.Load())
}

// Quantile estimates a quantile by linear interpolation inside the containing
// bucket. Bucketed quantiles are approximations and this one is no exception;
// it is here so the bench harness can print a p99 without a second dependency,
// not as a substitute for a real histogram backend.
func (h *Histogram) Quantile(q float64, values ...string) float64 {
	hv := h.slot(values)
	total := hv.total.Load()
	if total == 0 {
		return 0
	}
	target := q * float64(total)
	var cum float64
	prevBound := 0.0
	for i, b := range h.buckets {
		cum += float64(hv.counts[i].Load())
		if cum >= target {
			c := float64(hv.counts[i].Load())
			if c == 0 {
				return b
			}
			frac := (target - (cum - c)) / c
			return prevBound + frac*(b-prevBound)
		}
		prevBound = b
	}
	return h.buckets[len(h.buckets)-1]
}

func (h *Histogram) write(w io.Writer) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for _, k := range h.keys {
		hv := h.vals[k]
		inner := strings.TrimSuffix(strings.TrimPrefix(k, "{"), "}")
		var cum uint64
		for i, b := range h.buckets {
			cum += hv.counts[i].Load()
			fmt.Fprintf(w, "%s_bucket%s %d\n", h.n, withLE(inner, formatFloat(b)), cum)
		}
		cum += hv.counts[len(h.buckets)].Load()
		fmt.Fprintf(w, "%s_bucket%s %d\n", h.n, withLE(inner, "+Inf"), cum)
		fmt.Fprintf(w, "%s_sum%s %s\n", h.n, k, formatFloat(math.Float64frombits(hv.sum.Load())))
		fmt.Fprintf(w, "%s_count%s %d\n", h.n, k, hv.total.Load())
	}
}

func withLE(inner, le string) string {
	if inner == "" {
		return `{le="` + le + `"}`
	}
	return "{" + inner + `,le="` + le + `"}`
}

func formatFloat(f float64) string {
	if math.IsInf(f, 1) {
		return "+Inf"
	}
	if math.IsInf(f, -1) {
		return "-Inf"
	}
	if math.IsNaN(f) {
		return "NaN"
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}
