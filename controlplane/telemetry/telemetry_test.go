package telemetry_test

import (
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/telemetry"
)

func render(t *testing.T, r *telemetry.Registry) string {
	t.Helper()
	var b strings.Builder
	if _, err := r.WriteTo(&b); err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	return b.String()
}

func TestCounterExposition(t *testing.T) {
	r := telemetry.NewRegistry()
	c := telemetry.NewCounter(r, "requests_total", "Total requests.", "outcome")
	c.Inc("ok")
	c.Inc("ok")
	c.Add(3, "error")

	if got := c.Value("ok"); got != 2 {
		t.Fatalf("Value(ok) = %d, want 2", got)
	}
	out := render(t, r)
	for _, want := range []string{
		"# HELP requests_total Total requests.",
		"# TYPE requests_total counter",
		`requests_total{outcome="error"} 3`,
		`requests_total{outcome="ok"} 2`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("exposition missing %q:\n%s", want, out)
		}
	}
}

func TestUnlabelledCounterStillRenders(t *testing.T) {
	r := telemetry.NewRegistry()
	telemetry.NewCounter(r, "never_touched_total", "Never incremented.")
	out := render(t, r)
	if !strings.Contains(out, "never_touched_total 0") {
		t.Fatalf("a registered but unused counter should render as zero:\n%s", out)
	}
}

func TestGauge(t *testing.T) {
	r := telemetry.NewRegistry()
	g := telemetry.NewGauge(r, "in_flight", "In flight.")
	g.Set(7.5)
	if got := g.Value(); got != 7.5 {
		t.Fatalf("Value = %v, want 7.5", got)
	}
	g.Set(2)
	out := render(t, r)
	if !strings.Contains(out, "in_flight 2") {
		t.Fatalf("gauge did not update:\n%s", out)
	}
}

func TestHistogramBucketsAreCumulative(t *testing.T) {
	r := telemetry.NewRegistry()
	h := telemetry.NewHistogram(r, "latency_seconds", "Latency.", []float64{0.1, 1, 10})

	for _, v := range []float64{0.05, 0.5, 0.5, 5, 50} {
		h.Observe(v)
	}

	if got := h.Count(); got != 5 {
		t.Fatalf("Count = %d, want 5", got)
	}
	if got := h.Sum(); got != 56.05 {
		t.Fatalf("Sum = %v, want 56.05", got)
	}

	out := render(t, r)
	for _, want := range []string{
		`latency_seconds_bucket{le="0.1"} 1`,
		`latency_seconds_bucket{le="1"} 3`,
		`latency_seconds_bucket{le="10"} 4`,
		`latency_seconds_bucket{le="+Inf"} 5`,
		"latency_seconds_count 5",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("exposition missing %q:\n%s", want, out)
		}
	}
}

// TestHistogramBoundaryValue pins down the off-by-one that bucketed histograms
// always get wrong: an observation exactly on a boundary belongs in that
// bucket, because Prometheus buckets are "less than or equal".
func TestHistogramBoundaryValue(t *testing.T) {
	r := telemetry.NewRegistry()
	h := telemetry.NewHistogram(r, "b", "Boundaries.", []float64{1, 2})
	h.Observe(1)
	out := render(t, r)
	if !strings.Contains(out, `b_bucket{le="1"} 1`) {
		t.Fatalf("an observation of exactly 1 belongs in le=1:\n%s", out)
	}
}

func TestHistogramQuantile(t *testing.T) {
	r := telemetry.NewRegistry()
	h := telemetry.NewHistogram(r, "q", "Quantiles.", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	for i := 1; i <= 100; i++ {
		h.Observe(float64(i) / 10)
	}
	// The observations are 0.1 through 10.0 in steps of 0.1, so the true
	// median is about 5.05 and a bucketed estimate should land near it.
	p50 := h.Quantile(0.5)
	if p50 < 4 || p50 > 6 {
		t.Fatalf("p50 = %v, expected roughly 5", p50)
	}
	if p90 := h.Quantile(0.9); p90 < p50 {
		t.Fatalf("p90 %v is below p50 %v", p90, p50)
	}
	if h.Quantile(0.5) > h.Quantile(0.99) {
		t.Fatal("p50 exceeds p99")
	}
}

func TestLabelEscaping(t *testing.T) {
	r := telemetry.NewRegistry()
	c := telemetry.NewCounter(r, "escaped_total", "Escaping.", "path")
	c.Inc(`a"b\c`)
	out := render(t, r)
	if !strings.Contains(out, `escaped_total{path="a\"b\\c"} 1`) {
		t.Fatalf("label value was not escaped:\n%s", out)
	}
}

func TestDuplicateNamePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("registering the same metric name twice should panic at startup")
		}
	}()
	r := telemetry.NewRegistry()
	telemetry.NewCounter(r, "dup", "One.")
	telemetry.NewCounter(r, "dup", "Two.")
}

func TestHandler(t *testing.T) {
	r := telemetry.NewRegistry()
	telemetry.NewCounter(r, "served_total", "Served.").Inc()

	rec := httptest.NewRecorder()
	r.Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("Content-Type = %q", ct)
	}
	if !strings.Contains(rec.Body.String(), "served_total 1") {
		t.Fatalf("body:\n%s", rec.Body.String())
	}
}

func TestConcurrentUse(t *testing.T) {
	r := telemetry.NewRegistry()
	c := telemetry.NewCounter(r, "c_total", "C.", "k")
	g := telemetry.NewGauge(r, "g", "G.")
	h := telemetry.NewHistogram(r, "h", "H.", nil, "k")

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				c.Inc("a")
				g.Set(float64(j))
				h.Observe(0.01, "a")
			}
		}(i)
	}
	// Scrape concurrently with the writers; this is what a real scrape does.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			var b strings.Builder
			r.WriteTo(&b)
		}
	}()
	wg.Wait()

	if got := c.Value("a"); got != 16*200 {
		t.Fatalf("counter lost increments: %d, want %d", got, 16*200)
	}
	if got := h.Count("a"); got != 16*200 {
		t.Fatalf("histogram lost observations: %d", got)
	}
}
