package backend_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/backend"
)

func reqs(n int, prefix string) []backend.Request {
	out := make([]backend.Request, n)
	for i := range out {
		out[i] = backend.Request{
			ID:           string(rune('a' + i%26)),
			PrefixKey:    prefix,
			PromptTokens: 100,
			MaxTokens:    10,
		}
	}
	return out
}

func TestOneResponsePerRequest(t *testing.T) {
	s := backend.NewSim(backend.SimConfig{Name: "r0"})
	defer s.Close()

	in := reqs(5, "p")
	out, err := s.Infer(context.Background(), in)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(in) {
		t.Fatalf("got %d responses for %d requests", len(out), len(in))
	}
	for i := range out {
		if out[i].ID != in[i].ID {
			t.Fatalf("response %d is for %q, want %q", i, out[i].ID, in[i].ID)
		}
		if out[i].Replica != "r0" {
			t.Fatalf("response %d came from %q", i, out[i].Replica)
		}
	}
}

// TestBatchAmortisesLaunchOverhead is the property the whole batcher relies on.
// If a simulated backend did not have it, every batching test downstream would
// be measuring nothing.
func TestBatchAmortisesLaunchOverhead(t *testing.T) {
	cfg := backend.SimConfig{
		Name:            "r0",
		LaunchOverhead:  20 * time.Millisecond,
		PerTokenPrefill: time.Microsecond,
		PerTokenDecode:  time.Microsecond,
	}
	s := backend.NewSim(cfg)
	defer s.Close()

	single := time.Now()
	for i := 0; i < 8; i++ {
		if _, err := s.Infer(context.Background(), reqs(1, "")); err != nil {
			t.Fatal(err)
		}
	}
	sequential := time.Since(single)

	batched := time.Now()
	if _, err := s.Infer(context.Background(), reqs(8, "")); err != nil {
		t.Fatal(err)
	}
	oneBatch := time.Since(batched)

	if oneBatch >= sequential/2 {
		t.Fatalf("one batch of 8 took %v, eight singles took %v: launch overhead is not being amortised",
			oneBatch, sequential)
	}
	t.Logf("8 sequential: %v, 1 batch of 8: %v (%.1fx)", sequential, oneBatch,
		float64(sequential)/float64(oneBatch))
}

// TestPrefixCacheHits is what makes affinity routing worth implementing.
func TestPrefixCacheHits(t *testing.T) {
	s := backend.NewSim(backend.SimConfig{Name: "r0", CacheSlots: 2})
	defer s.Close()

	// First visit for a prefix is a miss, subsequent ones are hits.
	out, err := s.Infer(context.Background(), []backend.Request{{ID: "1", PrefixKey: "sys"}})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].CacheHit {
		t.Fatal("first request for a prefix reported a cache hit")
	}

	out, err = s.Infer(context.Background(), []backend.Request{{ID: "2", PrefixKey: "sys"}})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].CacheHit {
		t.Fatal("repeat request for the same prefix missed")
	}

	// Evict it by touching more distinct prefixes than there are slots.
	for _, k := range []string{"a", "b", "c"} {
		if _, err := s.Infer(context.Background(), []backend.Request{{ID: k, PrefixKey: k}}); err != nil {
			t.Fatal(err)
		}
	}
	out, err = s.Infer(context.Background(), []backend.Request{{ID: "3", PrefixKey: "sys"}})
	if err != nil {
		t.Fatal(err)
	}
	if out[0].CacheHit {
		t.Fatal("evicted prefix still reported a hit")
	}
}

func TestEmptyPrefixNeverHits(t *testing.T) {
	s := backend.NewSim(backend.SimConfig{Name: "r0"})
	defer s.Close()
	for i := 0; i < 3; i++ {
		out, err := s.Infer(context.Background(), []backend.Request{{ID: "x"}})
		if err != nil {
			t.Fatal(err)
		}
		if out[0].CacheHit {
			t.Fatal("a request with no prefix key reported a cache hit")
		}
	}
}

func TestUnhealthyAndClosed(t *testing.T) {
	s := backend.NewSim(backend.SimConfig{Name: "r0"})

	s.SetHealthy(false)
	if _, err := s.Infer(context.Background(), reqs(1, "")); !errors.Is(err, backend.ErrUnhealthy) {
		t.Fatalf("got %v, want ErrUnhealthy", err)
	}
	s.SetHealthy(true)
	if _, err := s.Infer(context.Background(), reqs(1, "")); err != nil {
		t.Fatal(err)
	}

	s.Close()
	if s.Healthy() {
		t.Fatal("a closed backend reports healthy")
	}
}

func TestMaxBatch(t *testing.T) {
	s := backend.NewSim(backend.SimConfig{Name: "r0", MaxBatch: 4})
	defer s.Close()
	if _, err := s.Infer(context.Background(), reqs(5, "")); !errors.Is(err, backend.ErrOverloaded) {
		t.Fatalf("got %v, want ErrOverloaded", err)
	}
}

func TestContextCancellation(t *testing.T) {
	s := backend.NewSim(backend.SimConfig{Name: "r0", LaunchOverhead: time.Second})
	defer s.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	start := time.Now()
	if _, err := s.Infer(ctx, reqs(1, "")); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want DeadlineExceeded", err)
	}
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Fatalf("cancellation took %v to take effect", d)
	}
}

func TestFailureInjectionIsDeterministic(t *testing.T) {
	run := func() int {
		s := backend.NewSim(backend.SimConfig{Name: "r0", FailureRate: 0.5, Seed: 42, LaunchOverhead: time.Microsecond})
		defer s.Close()
		fails := 0
		for i := 0; i < 50; i++ {
			if _, err := s.Infer(context.Background(), reqs(1, "")); err != nil {
				fails++
			}
		}
		return fails
	}
	a, b := run(), run()
	if a != b {
		t.Fatalf("same seed produced %d and %d failures", a, b)
	}
	if a == 0 || a == 50 {
		t.Fatalf("failure injection produced %d/50 failures, which is not a 0.5 rate", a)
	}
}

func TestRequestCost(t *testing.T) {
	cheap := backend.Request{PromptTokens: 100, MaxTokens: 1}
	pricey := backend.Request{PromptTokens: 100, MaxTokens: 100}
	if cheap.Cost() >= pricey.Cost() {
		t.Fatal("decode tokens must dominate the cost estimate")
	}
}
