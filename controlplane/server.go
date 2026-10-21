// Package controlplane wires admission, routing, batching, placement and
// telemetry into one service.
//
// This is the answer to the question the rest of the repository asks. Go loses
// the data plane - the measurements in pkg/ffi, pkg/kernel, pkg/memory and
// pkg/device are not close, and no amount of assembly closes them. What Go wins
// is everything around it: a hundred thousand concurrent connections, per
// request cancellation that actually propagates, timers that do not leak,
// graceful shutdown, and a static binary that starts in milliseconds and can be
// dropped into a distroless image.
//
// So the boundary is drawn here. Tensors go to a runtime built for tensors. The
// scheduling, the fleet state, the SLO enforcement and the observability stay
// in Go, where they are ordinary well-understood code rather than a fight with
// the language.
package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/admission"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/backend"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/batcher"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/router"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/telemetry"
	"github.com/udaykishore-resu/gpu-boundary/controlplane/topology"
)

// Errors returned by the server.
var (
	// ErrNoBackend means no replica has been registered.
	ErrNoBackend = errors.New("controlplane: no replicas registered")
	// ErrClosed means the server is shutting down.
	ErrClosed = errors.New("controlplane: closed")
	// ErrDuplicateReplica means a replica with that ID already exists.
	ErrDuplicateReplica = errors.New("controlplane: duplicate replica")
)

// BatchOptions tune the per-replica batcher.
type BatchOptions struct {
	MaxSize int
	MaxWait time.Duration
	Workers int
}

// Options configures a server.
type Options struct {
	Admission admission.Config
	Router    router.Config
	Batch     BatchOptions
	// Registry receives the server's metrics. A nil Registry gets a fresh one.
	Registry *telemetry.Registry
}

// Server is the control plane.
type Server struct {
	opts  Options
	lim   *admission.Limiter
	rtr   *router.Router
	fleet *topology.Fleet
	reg   *telemetry.Registry

	mu       sync.RWMutex
	backends map[string]backend.Backend
	batchers map[string]*batcher.Batcher[backend.Request, backend.Response]
	closed   bool

	m metrics
}

type metrics struct {
	requests    *telemetry.Counter
	rejected    *telemetry.Counter
	decisions   *telemetry.Counter
	cacheHits   *telemetry.Counter
	latency     *telemetry.Histogram
	batchSize   *telemetry.Histogram
	inFlight    *telemetry.Gauge
	queued      *telemetry.Gauge
	replicaLoad *telemetry.Gauge
	fleetUtil   *telemetry.Gauge
}

// New returns a server. Close must be called to release its goroutines.
func New(opts Options) *Server {
	if opts.Registry == nil {
		opts.Registry = telemetry.NewRegistry()
	}
	if opts.Batch.MaxSize <= 0 {
		opts.Batch.MaxSize = 16
	}
	if opts.Batch.MaxWait <= 0 {
		opts.Batch.MaxWait = 4 * time.Millisecond
	}
	if opts.Batch.Workers <= 0 {
		opts.Batch.Workers = 2
	}

	r := opts.Registry
	s := &Server{
		opts:     opts,
		lim:      admission.New(opts.Admission),
		rtr:      router.New(opts.Router),
		fleet:    topology.NewFleet(),
		reg:      r,
		backends: make(map[string]backend.Backend),
		batchers: make(map[string]*batcher.Batcher[backend.Request, backend.Response]),
		m: metrics{
			requests:  telemetry.NewCounter(r, "inference_requests_total", "Requests by outcome.", "outcome"),
			rejected:  telemetry.NewCounter(r, "inference_rejected_total", "Requests refused before execution, by cause.", "cause"),
			decisions: telemetry.NewCounter(r, "router_decisions_total", "Routing decisions by reason.", "reason"),
			cacheHits: telemetry.NewCounter(r, "inference_prefix_cache_hits_total", "Responses that reported a KV prefix cache hit.", "replica"),
			latency:   telemetry.NewHistogram(r, "inference_latency_seconds", "End to end latency as the control plane sees it.", nil, "replica"),
			batchSize: telemetry.NewHistogram(r, "inference_batch_size", "Requests per dispatched batch.",
				[]float64{1, 2, 4, 8, 16, 32, 64, 128}, "replica"),
			inFlight:    telemetry.NewGauge(r, "admission_in_flight", "Requests currently executing."),
			queued:      telemetry.NewGauge(r, "admission_queued", "Requests waiting for a slot."),
			replicaLoad: telemetry.NewGauge(r, "replica_in_flight", "In-flight requests per replica.", "replica"),
			fleetUtil:   telemetry.NewGauge(r, "fleet_memory_utilisation", "Fraction of fleet accelerator memory reserved."),
		},
	}
	return s
}

// Registry exposes the metric registry, so a caller can add its own metrics or
// serve them on a port of its choosing.
func (s *Server) Registry() *telemetry.Registry { return s.reg }

// Fleet exposes the topology, so devices can be registered and models placed.
func (s *Server) Fleet() *topology.Fleet { return s.fleet }

// AddReplica registers a backend and gives it its own batcher.
//
// Per-replica batching rather than one global batcher is the important detail:
// a batch is only useful if every request in it runs on the same accelerator
// with the same weights resident, so the batch boundary has to sit downstream
// of the routing decision, not upstream of it.
func (s *Server) AddReplica(b backend.Backend, rep router.Replica) error {
	if rep.ID == "" {
		rep.ID = b.Name()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	if _, exists := s.backends[rep.ID]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateReplica, rep.ID)
	}

	id := rep.ID
	bat, err := batcher.New(batcher.Config[backend.Request, backend.Response]{
		MaxSize: s.opts.Batch.MaxSize,
		MaxWait: s.opts.Batch.MaxWait,
		Workers: s.opts.Batch.Workers,
		Process: func(ctx context.Context, reqs []backend.Request) ([]backend.Response, error) {
			s.m.batchSize.Observe(float64(len(reqs)), id)
			return b.Infer(ctx, reqs)
		},
	})
	if err != nil {
		return err
	}

	if err := s.rtr.Add(rep); err != nil {
		bat.Close()
		return err
	}
	s.backends[id] = b
	s.batchers[id] = bat
	return nil
}

// RemoveReplica drains and deregisters a replica.
func (s *Server) RemoveReplica(id string) error {
	s.mu.Lock()
	bat, ok := s.batchers[id]
	delete(s.batchers, id)
	delete(s.backends, id)
	s.mu.Unlock()

	s.rtr.Remove(id)
	if !ok {
		return fmt.Errorf("%w: %s", router.ErrUnknownReplica, id)
	}
	return bat.Close()
}

// Infer serves one request.
//
// The order is admission, then routing, then batching, and it is not
// interchangeable. Admission first, so an overloaded system spends nothing on
// work it will refuse. Routing before batching, so a batch is homogeneous in
// replica. Batching last, so the accelerator sees as few submissions as the
// latency budget allows.
func (s *Server) Infer(ctx context.Context, req backend.Request) (backend.Response, error) {
	start := time.Now()

	s.mu.RLock()
	closed := s.closed
	n := len(s.batchers)
	s.mu.RUnlock()
	if closed {
		return backend.Response{}, ErrClosed
	}
	if n == 0 {
		return backend.Response{}, ErrNoBackend
	}

	release, err := s.lim.Acquire(ctx, req.Deadline)
	if err != nil {
		s.m.rejected.Inc(causeOf(err))
		s.m.requests.Inc("rejected")
		return backend.Response{}, err
	}
	defer release()
	s.updateGauges()

	decision, done, err := s.rtr.Pick(req.PrefixKey)
	if err != nil {
		s.m.rejected.Inc("no-replica")
		s.m.requests.Inc("rejected")
		return backend.Response{}, err
	}
	s.m.decisions.Inc(string(decision.Reason))

	s.mu.RLock()
	bat := s.batchers[decision.Replica]
	s.mu.RUnlock()
	if bat == nil {
		done(false)
		s.m.requests.Inc("error")
		return backend.Response{}, fmt.Errorf("%w: %s", router.ErrUnknownReplica, decision.Replica)
	}

	resp, err := bat.Submit(ctx, req)
	done(resp.CacheHit)

	elapsed := time.Since(start)
	s.m.latency.Observe(elapsed.Seconds(), decision.Replica)
	s.updateGauges()

	if err != nil {
		s.m.requests.Inc("error")
		return backend.Response{}, err
	}
	if resp.CacheHit {
		s.m.cacheHits.Inc(decision.Replica)
	}
	s.m.requests.Inc("ok")

	resp.Replica = decision.Replica
	resp.Latency = elapsed
	return resp, nil
}

func causeOf(err error) string {
	switch {
	case errors.Is(err, admission.ErrDeadlineUnreachable):
		return "deadline-unreachable"
	case errors.Is(err, admission.ErrQueueFull):
		return "queue-full"
	case errors.Is(err, admission.ErrClosed):
		return "closed"
	case errors.Is(err, context.Canceled):
		return "client-cancelled"
	case errors.Is(err, context.DeadlineExceeded):
		return "client-timeout"
	default:
		return "other"
	}
}

func (s *Server) updateGauges() {
	st := s.lim.Stats()
	s.m.inFlight.Set(float64(st.InFlight))
	s.m.queued.Set(float64(st.Queued))
	for _, r := range s.rtr.Snapshot() {
		s.m.replicaLoad.Set(float64(r.InFlight), r.ID)
	}
	s.m.fleetUtil.Set(s.fleet.Utilisation())
}

// Status is the JSON shape served at /v1/status.
type Status struct {
	Replicas   []router.ReplicaState    `json:"replicas"`
	Admission  admission.Stats          `json:"admission"`
	Decisions  map[string]uint64        `json:"routing_decisions"`
	Bindings   int                      `json:"prefix_bindings"`
	Devices    []topology.DeviceStatus  `json:"devices"`
	Placements []topology.Placement     `json:"placements"`
	Batching   map[string]batcher.Stats `json:"batching"`
}

// Status snapshots the whole control plane.
func (s *Server) Status() Status {
	s.mu.RLock()
	ids := make([]string, 0, len(s.batchers))
	for id := range s.batchers {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	bs := make(map[string]batcher.Stats, len(ids))
	for _, id := range ids {
		bs[id] = s.batchers[id].Stats()
	}
	s.mu.RUnlock()

	dec := make(map[string]uint64)
	for k, v := range s.rtr.Decisions() {
		dec[string(k)] = v
	}

	return Status{
		Replicas:   s.rtr.Snapshot(),
		Admission:  s.lim.Stats(),
		Decisions:  dec,
		Bindings:   s.rtr.Bindings(),
		Devices:    s.fleet.Status(),
		Placements: s.fleet.Placements(),
		Batching:   bs,
	}
}

// Ready reports whether at least one replica is healthy.
func (s *Server) Ready() bool {
	for _, r := range s.rtr.Snapshot() {
		if r.Healthy {
			return true
		}
	}
	return false
}

// inferRequest is the wire shape accepted by /v1/infer.
type inferRequest struct {
	ID           string `json:"id"`
	Model        string `json:"model"`
	PrefixKey    string `json:"prefix_key"`
	PromptTokens int    `json:"prompt_tokens"`
	MaxTokens    int    `json:"max_tokens"`
	TimeoutMS    int    `json:"timeout_ms"`
}

type inferResponse struct {
	ID           string `json:"id"`
	Replica      string `json:"replica"`
	OutputTokens int    `json:"output_tokens"`
	CacheHit     bool   `json:"cache_hit"`
	LatencyMS    int64  `json:"latency_ms"`
}

type errorResponse struct {
	Error string `json:"error"`
	Cause string `json:"cause"`
}

// Handler returns the HTTP surface: /v1/infer, /v1/status, /metrics, /healthz
// and /readyz.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/infer", func(w http.ResponseWriter, r *http.Request) {
		var in inferRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&in); err != nil {
			writeJSON(w, http.StatusBadRequest, errorResponse{Error: err.Error(), Cause: "bad-request"})
			return
		}
		if in.MaxTokens <= 0 {
			in.MaxTokens = 32
		}
		if in.PromptTokens < 0 {
			in.PromptTokens = 0
		}

		ctx := r.Context()
		req := backend.Request{
			ID:           in.ID,
			Model:        in.Model,
			PrefixKey:    in.PrefixKey,
			PromptTokens: in.PromptTokens,
			MaxTokens:    in.MaxTokens,
		}
		if in.TimeoutMS > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, time.Duration(in.TimeoutMS)*time.Millisecond)
			defer cancel()
			req.Deadline = time.Now().Add(time.Duration(in.TimeoutMS) * time.Millisecond)
		}

		resp, err := s.Infer(ctx, req)
		if err != nil {
			writeJSON(w, statusFor(err), errorResponse{Error: err.Error(), Cause: causeOf(err)})
			return
		}
		writeJSON(w, http.StatusOK, inferResponse{
			ID:           resp.ID,
			Replica:      resp.Replica,
			OutputTokens: resp.OutputTokens,
			CacheHit:     resp.CacheHit,
			LatencyMS:    resp.Latency.Milliseconds(),
		})
	})

	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.Status())
	})

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok\n"))
	})

	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if !s.Ready() {
			http.Error(w, "no healthy replica", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ready\n"))
	})

	mux.Handle("GET /metrics", s.reg.Handler())
	return mux
}

// statusFor maps a control plane error onto an HTTP status that means something
// to a client and to an autoscaler. 429 says retry with backoff; 503 says this
// instance has nothing to serve you; 504 says you ran out of your own time.
func statusFor(err error) int {
	switch {
	case errors.Is(err, admission.ErrQueueFull),
		errors.Is(err, admission.ErrDeadlineUnreachable):
		return http.StatusTooManyRequests
	case errors.Is(err, ErrNoBackend),
		errors.Is(err, router.ErrNoReplicas),
		errors.Is(err, router.ErrAllUnhealthy),
		errors.Is(err, ErrClosed),
		errors.Is(err, admission.ErrClosed):
		return http.StatusServiceUnavailable
	case errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout
	case errors.Is(err, context.Canceled):
		// The client is gone; the status code is for the access log only.
		return 499
	default:
		return http.StatusInternalServerError
	}
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.Encode(v)
}

// SetReplicaHealth marks a replica in or out of rotation.
func (s *Server) SetReplicaHealth(id string, healthy bool) error {
	return s.rtr.SetHealthy(id, healthy)
}

// Close stops admitting work, drains the batchers and closes every backend.
func (s *Server) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	bats := make([]*batcher.Batcher[backend.Request, backend.Response], 0, len(s.batchers))
	for _, b := range s.batchers {
		bats = append(bats, b)
	}
	bks := make([]backend.Backend, 0, len(s.backends))
	for _, b := range s.backends {
		bks = append(bks, b)
	}
	s.mu.Unlock()

	s.lim.Close()

	var errs []error
	for _, b := range bats {
		if err := b.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	for _, b := range bks {
		if err := b.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
