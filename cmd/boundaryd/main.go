// Command boundaryd runs the control plane.
//
// It serves the HTTP surface against simulated inference replicas, so the whole
// thing can be exercised on a laptop with no GPU in it. Replacing the simulated
// backend with a real one - a vLLM or TensorRT-LLM endpoint over HTTP - means
// implementing backend.Backend and changing one line in this file; nothing else
// in the control plane knows or cares what is behind the interface.
//
// Usage:
//
//	go run ./cmd/boundaryd
//	go run ./cmd/boundaryd -replicas=8 -addr=:9090 -load=200
//
// Endpoints:
//
//	POST /v1/infer    {"id","model","prefix_key","prompt_tokens","max_tokens","timeout_ms"}
//	GET  /v1/status   fleet, routing, admission and batching state
//	GET  /metrics     Prometheus text exposition
//	GET  /healthz     liveness
//	GET  /readyz      readiness: at least one healthy replica
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/admission"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/backend"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/router"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/topology"
)

func main() {
	var (
		addr        = flag.String("addr", ":8080", "listen address")
		replicas    = flag.Int("replicas", 4, "number of simulated replicas")
		model       = flag.String("model", "llama-3.1-8b", "model name to advertise")
		maxInFlight = flag.Int("max-inflight", 128, "admission concurrency ceiling")
		maxQueue    = flag.Int("max-queue", 512, "admission queue depth")
		batchSize   = flag.Int("batch-size", 24, "maximum requests per batch")
		batchWait   = flag.Duration("batch-wait", 4*time.Millisecond, "maximum time the first request in a batch waits")
		affinity    = flag.Float64("affinity", 2.0, "how much extra load an affine replica may carry before the router defects (1.0 disables affinity)")
		load        = flag.Int("load", 0, "if positive, generate this many synthetic requests per second against the local server")
		sessions    = flag.Int("sessions", 64, "distinct prompt prefixes used by the load generator")
		verbose     = flag.Bool("v", false, "debug logging")
	)
	flag.Parse()

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	srv := controlplane.New(controlplane.Options{
		Admission: admission.Config{
			MaxInFlight:     *maxInFlight,
			MaxQueue:        *maxQueue,
			ServiceEstimate: 40 * time.Millisecond,
		},
		Router: router.Config{
			AffinityFactor: *affinity,
			AffinityTTL:    5 * time.Minute,
		},
		Batch: controlplane.BatchOptions{
			MaxSize: *batchSize,
			MaxWait: *batchWait,
			Workers: 2,
		},
	})

	if err := buildFleet(srv, *replicas, *model, log); err != nil {
		log.Error("building fleet", "err", err)
		os.Exit(1)
	}

	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Info("listening", "addr", *addr, "replicas", *replicas, "model", *model,
			"batch_size", *batchSize, "batch_wait", batchWait.String(), "affinity", *affinity)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("http server", "err", err)
			stop()
		}
	}()

	if *load > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			generateLoad(ctx, srv, *load, *sessions, *model, log)
		}()
	}

	<-ctx.Done()
	log.Info("shutting down")

	// Stop accepting connections first, then drain the control plane. The
	// order matters: closing the control plane first would fail requests that
	// are already in flight and would have completed fine.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Warn("http shutdown", "err", err)
	}
	if err := srv.Close(); err != nil {
		log.Warn("control plane shutdown", "err", err)
	}
	wg.Wait()

	st := srv.Status()
	log.Info("final state",
		"admitted", st.Admission.Admitted,
		"shed", st.Admission.Shed,
		"rejected", st.Admission.Rejected,
		"routing", fmt.Sprint(st.Decisions),
		"prefix_bindings", st.Bindings)
}

// buildFleet registers devices, places one replica per device and wires each
// placement to a simulated backend. On a real deployment the device inventory
// would come from a node agent and the placements from a scheduler; the shapes
// are the same.
func buildFleet(srv *controlplane.Server, n int, model string, log *slog.Logger) error {
	fleet := srv.Fleet()

	for i := 0; i < n; i++ {
		host := fmt.Sprintf("node-%d", i/4)
		if err := fleet.AddDevice(topology.Device{
			ID:            fmt.Sprintf("%s-gpu%d", host, i%4),
			Host:          host,
			Model:         "H100-80GB",
			MemoryMB:      80_000,
			Domain:        fmt.Sprintf("%s-nvl0", host),
			Link:          topology.LinkNVLink,
			ComputeTFlops: 989,
		}); err != nil {
			return err
		}
	}

	for i := 0; i < n; i++ {
		id := fmt.Sprintf("replica-%d", i)
		if _, err := fleet.Place(topology.Request{
			ID:       id,
			Model:    model,
			MemoryMB: 24_000,
		}); err != nil {
			return fmt.Errorf("placing %s: %w", id, err)
		}

		sim := backend.NewSim(backend.SimConfig{
			Name:             id,
			LaunchOverhead:   3 * time.Millisecond,
			PerTokenPrefill:  600 * time.Nanosecond,
			PerTokenDecode:   9 * time.Microsecond,
			CacheHitDiscount: 0.9,
			CacheSlots:       32,
			MaxBatch:         256,
			Seed:             int64(i),
		})
		if err := srv.AddReplica(sim, router.Replica{
			ID:            id,
			MaxConcurrent: 512,
			Weight:        1,
		}); err != nil {
			return err
		}
		log.Debug("replica ready", "id", id)
	}

	log.Info("fleet built", "devices", len(fleet.Status()), "placements", len(fleet.Placements()),
		"memory_utilisation", fmt.Sprintf("%.0f%%", fleet.Utilisation()*100))
	return nil
}

// generateLoad drives the control plane directly rather than over HTTP, so the
// numbers on /metrics describe the control plane and not the loopback network.
func generateLoad(ctx context.Context, srv *controlplane.Server, rps, sessions int, model string, log *slog.Logger) {
	interval := time.Second / time.Duration(rps)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	var seq int

	report := time.NewTicker(5 * time.Second)
	defer report.Stop()

	for {
		select {
		case <-ctx.Done():
			return

		case <-report.C:
			st := srv.Status()
			var served, hits uint64
			for _, r := range st.Replicas {
				served += r.Served
				hits += r.CacheHits
			}
			hitRate := 0.0
			if served > 0 {
				hitRate = float64(hits) / float64(served)
			}
			log.Info("load",
				"admitted", st.Admission.Admitted,
				"shed", st.Admission.Shed,
				"in_flight", st.Admission.InFlight,
				"queued", st.Admission.Queued,
				"prefix_hit_rate", fmt.Sprintf("%.0f%%", hitRate*100),
				"routing", fmt.Sprint(st.Decisions))

		case <-ticker.C:
			seq++
			// Zipf-ish session selection: a few conversations are hot, most
			// are cold, which is what real traffic looks like and what makes
			// prefix affinity worth anything.
			sess := int(rng.ExpFloat64() * float64(sessions) / 3)
			if sess >= sessions {
				sess = sessions - 1
			}
			req := backend.Request{
				ID:           fmt.Sprintf("gen-%d", seq),
				Model:        model,
				PrefixKey:    fmt.Sprintf("session-%d", sess),
				PromptTokens: 256 + rng.Intn(1792),
				MaxTokens:    16 + rng.Intn(112),
				Deadline:     time.Now().Add(2 * time.Second),
			}
			go func() {
				reqCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
				defer cancel()
				if _, err := srv.Infer(reqCtx, req); err != nil {
					log.Debug("request failed", "id", req.ID, "err", err)
				}
			}()
		}
	}
}
