// Package backend is the seam between the Go control plane and whatever
// actually runs the model.
//
// This is the load-bearing design decision of the whole repository. Everything
// measured in pkg/ffi, pkg/kernel and pkg/memory says the same thing: Go should
// not be executing the tensors. So the control plane does not try. It owns
// admission, batching, routing, placement and observability, and it delegates
// the forward pass over a network boundary to a runtime written in the language
// that owns that problem.
//
// A network hop sounds expensive until you price it against the alternative. It
// is tens of microseconds, it is amortised over a batch, and it buys the
// ability to run the data plane as a separate process with its own lifecycle,
// its own memory, and its own crash domain.
package backend

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Errors returned by backends.
var (
	// ErrUnhealthy means the backend is not currently accepting work.
	ErrUnhealthy = errors.New("backend: unhealthy")
	// ErrOverloaded means the backend rejected the batch rather than queue it.
	ErrOverloaded = errors.New("backend: overloaded")
)

// Request is one inference request as the control plane sees it.
//
// Note what is absent: no tensors, no tokens, no logits. The control plane
// routes on metadata and never touches model state, which is exactly why it can
// be written in Go without paying for any of the limitations this repository
// measures.
type Request struct {
	// ID is unique per request and appears in logs and metrics.
	ID string
	// Model selects the served model.
	Model string
	// PrefixKey identifies a shared prompt prefix - a system prompt, a
	// few-shot preamble, a conversation history. Requests with the same
	// PrefixKey can reuse each other's KV cache if they land on the same
	// replica, which is the entire reason the router exists.
	PrefixKey string
	// PromptTokens and MaxTokens size the work.
	PromptTokens int
	MaxTokens    int
	// Deadline is the caller's SLO. Admission uses it to shed early rather
	// than accept work that cannot finish in time.
	Deadline time.Time
}

// Cost estimates the relative expense of a request. Prefill is roughly linear
// in prompt length and decode is linear in generated tokens, with decode much
// more expensive per token because it is memory-bandwidth bound.
func (r Request) Cost() int {
	return r.PromptTokens + 8*r.MaxTokens
}

// Response is the result of one request.
type Response struct {
	ID           string
	Replica      string
	OutputTokens int
	CacheHit     bool
	Latency      time.Duration
	Err          error
}

// Backend executes a batch of requests.
//
// The interface is batch-shaped rather than request-shaped on purpose: every
// serious inference runtime wants batches, and an interface that accepts one
// request at a time quietly forbids the single most important optimisation in
// the system.
type Backend interface {
	// Name identifies the replica.
	Name() string
	// Infer runs a batch. The returned slice is parallel to the input.
	Infer(ctx context.Context, reqs []Request) ([]Response, error)
	// Healthy reports readiness.
	Healthy() bool
	// Close releases the backend.
	Close() error
}

// SimConfig describes a simulated runtime.
type SimConfig struct {
	// Name of the replica.
	Name string
	// LaunchOverhead is the fixed cost of executing a batch, regardless of
	// its size. This is the term batching amortises, and it is the reason a
	// batch of 32 is not 32 times slower than a batch of 1.
	LaunchOverhead time.Duration
	// PerTokenPrefill and PerTokenDecode price the actual work.
	PerTokenPrefill time.Duration
	PerTokenDecode  time.Duration
	// CacheHitDiscount is the fraction of prefill time saved when a request
	// arrives on a replica that already holds its prefix.
	CacheHitDiscount float64
	// CacheSlots is how many distinct prefixes the replica retains.
	CacheSlots int
	// MaxBatch rejects batches larger than this.
	MaxBatch int
	// FailureRate injects errors, so the control plane's error paths are
	// exercised by tests rather than only by production.
	FailureRate float64
	// Seed makes failure injection reproducible.
	Seed int64
}

func (c SimConfig) withDefaults() SimConfig {
	if c.Name == "" {
		c.Name = "sim"
	}
	if c.LaunchOverhead == 0 {
		c.LaunchOverhead = 2 * time.Millisecond
	}
	if c.PerTokenPrefill == 0 {
		c.PerTokenPrefill = 2 * time.Microsecond
	}
	if c.PerTokenDecode == 0 {
		c.PerTokenDecode = 20 * time.Microsecond
	}
	if c.CacheHitDiscount == 0 {
		c.CacheHitDiscount = 0.9
	}
	if c.CacheSlots == 0 {
		c.CacheSlots = 8
	}
	if c.MaxBatch == 0 {
		c.MaxBatch = 64
	}
	return c
}

// Sim is a deterministic stand-in for a real runtime.
//
// It models the two behaviours the control plane is designed around: a fixed
// per-batch launch overhead, and a KV cache whose hit rate depends on whether
// the router sent a prefix back to the replica that already has it. Without
// those two properties, batching and affinity routing would be untestable
// without a GPU, and code that is only testable on hardware you do not have is
// code that does not get tested.
type Sim struct {
	cfg SimConfig

	mu     sync.Mutex
	cache  []string // most-recently-used prefix keys
	rng    *rand.Rand
	closed bool

	batches  atomic.Uint64
	requests atomic.Uint64
	hits     atomic.Uint64
	healthy  atomic.Bool
}

// NewSim returns a simulated backend.
func NewSim(cfg SimConfig) *Sim {
	cfg = cfg.withDefaults()
	s := &Sim{
		cfg: cfg,
		rng: rand.New(rand.NewSource(cfg.Seed)),
	}
	s.healthy.Store(true)
	return s
}

// Name identifies the replica.
func (s *Sim) Name() string { return s.cfg.Name }

// Healthy reports readiness.
func (s *Sim) Healthy() bool { return s.healthy.Load() && !s.isClosed() }

// SetHealthy flips readiness, so tests can exercise failover.
func (s *Sim) SetHealthy(v bool) { s.healthy.Store(v) }

func (s *Sim) isClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

// Counters reports batches executed, requests served and cache hits.
func (s *Sim) Counters() (batches, requests, hits uint64) {
	return s.batches.Load(), s.requests.Load(), s.hits.Load()
}

// Infer executes a batch, sleeping for the modelled duration.
func (s *Sim) Infer(ctx context.Context, reqs []Request) ([]Response, error) {
	if len(reqs) == 0 {
		return nil, nil
	}
	if !s.Healthy() {
		return nil, fmt.Errorf("%s: %w", s.cfg.Name, ErrUnhealthy)
	}
	if len(reqs) > s.cfg.MaxBatch {
		return nil, fmt.Errorf("%s: %w: batch of %d exceeds %d",
			s.cfg.Name, ErrOverloaded, len(reqs), s.cfg.MaxBatch)
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, fmt.Errorf("%s: %w", s.cfg.Name, ErrUnhealthy)
	}
	fail := s.cfg.FailureRate > 0 && s.rng.Float64() < s.cfg.FailureRate
	hits := make([]bool, len(reqs))
	for i, r := range reqs {
		hits[i] = s.touchLocked(r.PrefixKey)
	}
	s.mu.Unlock()

	if fail {
		return nil, fmt.Errorf("%s: injected failure", s.cfg.Name)
	}

	// One launch overhead for the whole batch: the property the batcher exists
	// to exploit.
	work := s.cfg.LaunchOverhead
	for i, r := range reqs {
		prefill := time.Duration(r.PromptTokens) * s.cfg.PerTokenPrefill
		if hits[i] {
			prefill = time.Duration(float64(prefill) * (1 - s.cfg.CacheHitDiscount))
		}
		work += prefill + time.Duration(r.MaxTokens)*s.cfg.PerTokenDecode
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(work):
	}

	s.batches.Add(1)
	s.requests.Add(uint64(len(reqs)))

	out := make([]Response, len(reqs))
	for i, r := range reqs {
		if hits[i] {
			s.hits.Add(1)
		}
		out[i] = Response{
			ID:           r.ID,
			Replica:      s.cfg.Name,
			OutputTokens: r.MaxTokens,
			CacheHit:     hits[i],
			Latency:      work,
		}
	}
	return out, nil
}

// touchLocked records a prefix as recently used and reports whether it was
// already resident. This is a crude LRU, which is the right fidelity: the
// control plane only needs to know that affinity pays, not by exactly how much.
func (s *Sim) touchLocked(key string) bool {
	if key == "" {
		return false
	}
	for i, k := range s.cache {
		if k == key {
			copy(s.cache[1:i+1], s.cache[0:i])
			s.cache[0] = key
			return true
		}
	}
	s.cache = append([]string{key}, s.cache...)
	if len(s.cache) > s.cfg.CacheSlots {
		s.cache = s.cache[:s.cfg.CacheSlots]
	}
	return false
}

// HitRate reports the cache hit rate observed so far.
func (s *Sim) HitRate() float64 {
	reqs := s.requests.Load()
	if reqs == 0 {
		return math.NaN()
	}
	return float64(s.hits.Load()) / float64(reqs)
}

// Close marks the backend closed.
func (s *Sim) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}
