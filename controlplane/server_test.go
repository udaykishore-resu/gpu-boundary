package controlplane_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/admission"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/backend"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/router"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/topology"
)

func newServer(t *testing.T, replicas int, opts controlplane.Options) (*controlplane.Server, []*backend.Sim) {
	t.Helper()
	s := controlplane.New(opts)
	t.Cleanup(func() { s.Close() })

	sims := make([]*backend.Sim, replicas)
	for i := 0; i < replicas; i++ {
		sim := backend.NewSim(backend.SimConfig{
			Name:            fmt.Sprintf("r%d", i),
			LaunchOverhead:  2 * time.Millisecond,
			PerTokenPrefill: 500 * time.Nanosecond,
			PerTokenDecode:  time.Microsecond,
			CacheSlots:      16,
			MaxBatch:        256,
		})
		sims[i] = sim
		if err := s.AddReplica(sim, router.Replica{ID: sim.Name(), MaxConcurrent: 1024}); err != nil {
			t.Fatal(err)
		}
	}
	return s, sims
}

func TestNoReplicas(t *testing.T) {
	s := controlplane.New(controlplane.Options{})
	defer s.Close()
	if _, err := s.Infer(context.Background(), backend.Request{ID: "x"}); !errors.Is(err, controlplane.ErrNoBackend) {
		t.Fatalf("got %v, want ErrNoBackend", err)
	}
}

func TestEndToEnd(t *testing.T) {
	s, _ := newServer(t, 3, controlplane.Options{
		Admission: admission.Config{MaxInFlight: 64, MaxQueue: 512},
		Batch:     controlplane.BatchOptions{MaxSize: 16, MaxWait: 3 * time.Millisecond, Workers: 2},
	})

	const n = 300
	var wg sync.WaitGroup
	errs := make([]error, n)
	resps := make([]backend.Response, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			resps[i], errs[i] = s.Infer(context.Background(), backend.Request{
				ID:           fmt.Sprintf("req-%d", i),
				Model:        "llama-8b",
				PrefixKey:    fmt.Sprintf("session-%d", i%10),
				PromptTokens: 512,
				MaxTokens:    16,
			})
		}(i)
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Fatalf("request %d: %v", i, errs[i])
		}
		if resps[i].ID != fmt.Sprintf("req-%d", i) {
			t.Fatalf("request %d got a response for %q: responses are crossed", i, resps[i].ID)
		}
		if resps[i].Replica == "" {
			t.Fatalf("request %d has no replica recorded", i)
		}
	}

	st := s.Status()
	t.Logf("routing decisions: %v", st.Decisions)
	for id, b := range st.Batching {
		t.Logf("replica %s: %d batches, avg %.2f req/batch, max %d", id, b.Batches, b.AvgBatchSize, b.MaxObserved)
		if b.AvgBatchSize <= 1 {
			t.Errorf("replica %s never batched (avg %.2f)", id, b.AvgBatchSize)
		}
	}
}

// TestPrefixAffinityRaisesCacheHits is the end-to-end version of the routing
// argument. Ten sessions across three replicas should converge so that each
// session keeps hitting the replica that already holds its prefix.
func TestPrefixAffinityRaisesCacheHits(t *testing.T) {
	s, sims := newServer(t, 3, controlplane.Options{
		Admission: admission.Config{MaxInFlight: 8, MaxQueue: 512},
		Batch:     controlplane.BatchOptions{MaxSize: 8, MaxWait: 2 * time.Millisecond, Workers: 1},
	})

	for round := 0; round < 20; round++ {
		var wg sync.WaitGroup
		for sess := 0; sess < 10; sess++ {
			wg.Add(1)
			go func(sess int) {
				defer wg.Done()
				s.Infer(context.Background(), backend.Request{
					ID:           fmt.Sprintf("s%d-r%d", sess, round),
					PrefixKey:    fmt.Sprintf("session-%d", sess),
					PromptTokens: 1024,
					MaxTokens:    4,
				})
			}(sess)
		}
		wg.Wait()
	}

	var total, hits uint64
	for _, sim := range sims {
		_, reqs, h := sim.Counters()
		total += reqs
		hits += h
	}
	if total == 0 {
		t.Fatal("no requests reached a backend")
	}
	rate := float64(hits) / float64(total)
	t.Logf("prefix cache hit rate: %.1f%% (%d/%d)", rate*100, hits, total)
	if rate < 0.5 {
		t.Errorf("hit rate %.2f is below 0.5: affinity routing is not paying off", rate)
	}
}

// TestSheddingUnderOverload checks that the system refuses work rather than
// queueing it past its deadline, and that it says why.
func TestSheddingUnderOverload(t *testing.T) {
	s, _ := newServer(t, 1, controlplane.Options{
		Admission: admission.Config{
			MaxInFlight:     1,
			MaxQueue:        4,
			ServiceEstimate: 100 * time.Millisecond,
			ShedMargin:      time.Millisecond,
		},
		Batch: controlplane.BatchOptions{MaxSize: 1, MaxWait: time.Millisecond, Workers: 1},
	})

	var wg sync.WaitGroup
	var mu sync.Mutex
	shed, queueFull, ok := 0, 0, 0

	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := s.Infer(context.Background(), backend.Request{
				ID:           fmt.Sprintf("r%d", i),
				PromptTokens: 64,
				MaxTokens:    8,
				Deadline:     time.Now().Add(5 * time.Millisecond),
			})
			mu.Lock()
			defer mu.Unlock()
			switch {
			case err == nil:
				ok++
			case errors.Is(err, admission.ErrDeadlineUnreachable):
				shed++
			case errors.Is(err, admission.ErrQueueFull):
				queueFull++
			}
		}(i)
	}
	wg.Wait()

	t.Logf("ok=%d shed=%d queue-full=%d", ok, shed, queueFull)
	if shed+queueFull == 0 {
		t.Fatal("an overloaded system accepted every request: nothing is shedding")
	}
	if ok+shed+queueFull != 40 {
		t.Fatalf("outcomes do not add up: %d+%d+%d != 40", ok, shed, queueFull)
	}
}

func TestUnhealthyReplicaIsAvoided(t *testing.T) {
	s, sims := newServer(t, 2, controlplane.Options{
		Batch: controlplane.BatchOptions{MaxSize: 4, MaxWait: time.Millisecond},
	})

	if err := s.SetReplicaHealth("r0", false); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		resp, err := s.Infer(context.Background(), backend.Request{
			ID: fmt.Sprintf("q%d", i), PrefixKey: fmt.Sprintf("p%d", i), MaxTokens: 1,
		})
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		if resp.Replica == "r0" {
			t.Fatal("routed to a replica marked unhealthy")
		}
	}
	if b, _, _ := sims[0].Counters(); b != 0 {
		t.Fatalf("unhealthy replica executed %d batches", b)
	}
}

func TestBackendErrorSurfaces(t *testing.T) {
	s := controlplane.New(controlplane.Options{
		Batch: controlplane.BatchOptions{MaxSize: 2, MaxWait: time.Millisecond},
	})
	defer s.Close()

	sim := backend.NewSim(backend.SimConfig{Name: "bad", FailureRate: 1.0, LaunchOverhead: time.Microsecond})
	if err := s.AddReplica(sim, router.Replica{ID: "bad"}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Infer(context.Background(), backend.Request{ID: "x", MaxTokens: 1}); err == nil {
		t.Fatal("expected the injected backend failure to reach the caller")
	}
}

func TestDuplicateReplica(t *testing.T) {
	s := controlplane.New(controlplane.Options{})
	defer s.Close()
	sim := backend.NewSim(backend.SimConfig{Name: "r0"})
	if err := s.AddReplica(sim, router.Replica{ID: "r0"}); err != nil {
		t.Fatal(err)
	}
	if err := s.AddReplica(sim, router.Replica{ID: "r0"}); !errors.Is(err, controlplane.ErrDuplicateReplica) {
		t.Fatalf("got %v, want ErrDuplicateReplica", err)
	}
}

func TestRemoveReplica(t *testing.T) {
	s, _ := newServer(t, 2, controlplane.Options{
		Batch: controlplane.BatchOptions{MaxSize: 2, MaxWait: time.Millisecond},
	})
	if err := s.RemoveReplica("r0"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		resp, err := s.Infer(context.Background(), backend.Request{ID: "x", MaxTokens: 1})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Replica == "r0" {
			t.Fatal("routed to a removed replica")
		}
	}
	if err := s.RemoveReplica("nope"); err == nil {
		t.Fatal("removing an unknown replica should error")
	}
}

func TestClosedServerRefuses(t *testing.T) {
	s, _ := newServer(t, 1, controlplane.Options{})
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Infer(context.Background(), backend.Request{ID: "x"}); !errors.Is(err, controlplane.ErrClosed) {
		t.Fatalf("got %v, want ErrClosed", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestHTTPSurface(t *testing.T) {
	s, _ := newServer(t, 2, controlplane.Options{
		Admission: admission.Config{MaxInFlight: 16, MaxQueue: 64},
		Batch:     controlplane.BatchOptions{MaxSize: 4, MaxWait: 2 * time.Millisecond},
	})
	if err := s.Fleet().AddDevice(topology.Device{
		ID: "gpu0", Host: "node", MemoryMB: 80_000, Domain: "nvl0", Link: topology.LinkNVLink,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Fleet().Place(topology.Request{ID: "p", Model: "llama-8b", MemoryMB: 20_000}); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	// Health and readiness.
	for _, path := range []string{"/healthz", "/readyz"} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s returned %d", path, resp.StatusCode)
		}
	}

	// Inference.
	body, _ := json.Marshal(map[string]any{
		"id": "http-1", "model": "llama-8b", "prefix_key": "sess",
		"prompt_tokens": 256, "max_tokens": 8, "timeout_ms": 5000,
	})
	resp, err := http.Post(srv.URL+"/v1/infer", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("infer returned %d", resp.StatusCode)
	}
	var out struct {
		ID      string `json:"id"`
		Replica string `json:"replica"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.ID != "http-1" || out.Replica == "" {
		t.Fatalf("unexpected response: %+v", out)
	}

	// Malformed input.
	bad, err := http.Post(srv.URL+"/v1/infer", "application/json", strings.NewReader("{"))
	if err != nil {
		t.Fatal(err)
	}
	bad.Body.Close()
	if bad.StatusCode != http.StatusBadRequest {
		t.Fatalf("malformed body returned %d, want 400", bad.StatusCode)
	}

	// Status and metrics.
	st, err := http.Get(srv.URL + "/v1/status")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Body.Close()
	var status controlplane.Status
	if err := json.NewDecoder(st.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	if len(status.Replicas) != 2 {
		t.Fatalf("status lists %d replicas, want 2", len(status.Replicas))
	}
	if len(status.Devices) != 1 || len(status.Placements) != 1 {
		t.Fatalf("status lost the fleet: %d devices, %d placements", len(status.Devices), len(status.Placements))
	}

	mt, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer mt.Body.Close()
	var buf bytes.Buffer
	buf.ReadFrom(mt.Body)
	for _, want := range []string{
		"inference_requests_total",
		"router_decisions_total",
		"inference_latency_seconds_bucket",
		"fleet_memory_utilisation",
	} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("metrics endpoint is missing %q", want)
		}
	}
}

func BenchmarkInfer(b *testing.B) {
	s := controlplane.New(controlplane.Options{
		Admission: admission.Config{MaxInFlight: 256, MaxQueue: 4096},
		Batch:     controlplane.BatchOptions{MaxSize: 32, MaxWait: time.Millisecond, Workers: 2},
	})
	defer s.Close()
	for i := 0; i < 4; i++ {
		sim := backend.NewSim(backend.SimConfig{
			Name:            fmt.Sprintf("r%d", i),
			LaunchOverhead:  500 * time.Microsecond,
			PerTokenPrefill: 100 * time.Nanosecond,
			PerTokenDecode:  200 * time.Nanosecond,
			MaxBatch:        256,
		})
		s.AddReplica(sim, router.Replica{ID: sim.Name(), MaxConcurrent: 4096})
	}

	b.ReportAllocs()
	b.ResetTimer()
	var n int64
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			i++
			_, err := s.Infer(context.Background(), backend.Request{
				ID:           "b",
				PrefixKey:    fmt.Sprintf("p%d", i%64),
				PromptTokens: 128,
				MaxTokens:    4,
			})
			if err != nil {
				b.Error(err)
				return
			}
		}
	})
	_ = n
}
