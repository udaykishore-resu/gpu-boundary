// Package router chooses which replica serves a request.
//
// For a stateless web service, load balancing is a solved and boring problem:
// pick the least loaded backend. Inference is not stateless. Every replica
// holds a KV cache, and a request whose prompt prefix is already resident on a
// replica skips most of its prefill work. Sending that request anywhere else
// throws the cache away and pays full price.
//
// So the routing objective is a genuine two-term trade-off:
//
//	minimise:  expected_latency = queue_wait(replica) + prefill(replica, prefix)
//
// Pure affinity funnels every request with a hot prefix onto one replica and
// queues behind it. Pure least-load ignores the cache and pays prefill every
// time. This router does neither: it prefers the affine replica while its load
// stays within a configurable factor of the least loaded one, and defects
// otherwise. Unseen prefixes are placed by rendezvous hashing so that the same
// prefix lands on the same replica across the fleet without any coordination.
//
// This is a scheduling problem over metadata. It never touches a tensor, which
// is why it belongs in the control plane and why Go is a good language to write
// it in.
package router

import (
	"errors"
	"fmt"
	"hash/fnv"
	"sort"
	"sync"
	"time"
)

// Errors returned by the router.
var (
	// ErrNoReplicas means nothing is registered.
	ErrNoReplicas = errors.New("router: no replicas registered")
	// ErrAllUnhealthy means every replica is marked unhealthy or at capacity.
	ErrAllUnhealthy = errors.New("router: no healthy replica available")
	// ErrUnknownReplica means the named replica is not registered.
	ErrUnknownReplica = errors.New("router: unknown replica")
)

// Replica describes one serving instance.
type Replica struct {
	// ID is the replica's stable name.
	ID string
	// MaxConcurrent caps outstanding requests on this replica.
	MaxConcurrent int
	// Weight scales capacity for heterogeneous fleets: an H100 replica and an
	// L4 replica should not receive the same share.
	Weight float64
}

// Config tunes the routing policy.
type Config struct {
	// AffinityFactor is how much extra load an affine replica is allowed to
	// carry before the router defects to a less loaded one. 1.0 disables
	// affinity entirely; a large value makes affinity absolute. Around 1.5 to
	// 2.5 is usually right, and it should be tuned against measured cache hit
	// rate, not chosen by taste.
	AffinityFactor float64
	// AffinityTTL is how long a prefix stays bound to a replica after its
	// last use, and should approximate how long the replica actually keeps
	// the KV blocks.
	AffinityTTL time.Duration
	// MaxAffinityEntries bounds the binding table.
	MaxAffinityEntries int
	// Now is injectable for tests.
	Now func() time.Time
}

func (c Config) withDefaults() Config {
	if c.AffinityFactor <= 0 {
		c.AffinityFactor = 2.0
	}
	if c.AffinityTTL <= 0 {
		c.AffinityTTL = 2 * time.Minute
	}
	if c.MaxAffinityEntries <= 0 {
		c.MaxAffinityEntries = 100_000
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return c
}

// Decision records why a replica was chosen. Routing decisions that cannot be
// explained cannot be debugged at three in the morning, so the reason is part
// of the return value rather than a log line.
type Decision struct {
	Replica string
	Reason  Reason
	// AffineTo is the replica the prefix was bound to, if any, even when the
	// router chose to defect from it.
	AffineTo string
	// Load is the chosen replica's in-flight count at decision time.
	Load int
}

// Reason explains a routing decision.
type Reason string

// Routing reasons.
const (
	// ReasonAffinity means the prefix was already bound to this replica and
	// the replica was not too loaded to use.
	ReasonAffinity Reason = "affinity"
	// ReasonDefect means a binding existed but the replica was overloaded.
	ReasonDefect Reason = "defect-overloaded"
	// ReasonRendezvous means the prefix was unseen and was placed by hash, so
	// that a future request with the same prefix lands here again.
	ReasonRendezvous Reason = "rendezvous"
	// ReasonLeastLoaded means no useful prefix information was available.
	ReasonLeastLoaded Reason = "least-loaded"
	// ReasonFailover means the preferred replica was unhealthy.
	ReasonFailover Reason = "failover"
)

type replicaState struct {
	Replica
	inFlight  int
	healthy   bool
	served    uint64
	rejected  uint64
	cacheHits uint64
}

// load returns the weight-normalised in-flight count, which is what makes
// comparison across a heterogeneous fleet meaningful.
func (r *replicaState) load() float64 {
	w := r.Weight
	if w <= 0 {
		w = 1
	}
	return float64(r.inFlight) / w
}

func (r *replicaState) available() bool {
	return r.healthy && (r.MaxConcurrent <= 0 || r.inFlight < r.MaxConcurrent)
}

type binding struct {
	replica string
	touched time.Time
}

// Router picks replicas.
type Router struct {
	cfg Config

	mu       sync.Mutex
	replicas map[string]*replicaState
	order    []string // sorted, for deterministic iteration
	affinity map[string]*binding

	decisions map[Reason]uint64
}

// New returns a router.
func New(cfg Config) *Router {
	return &Router{
		cfg:       cfg.withDefaults(),
		replicas:  make(map[string]*replicaState),
		affinity:  make(map[string]*binding),
		decisions: make(map[Reason]uint64),
	}
}

// Add registers a replica, or updates one already registered.
func (r *Router) Add(rep Replica) error {
	if rep.ID == "" {
		return errors.New("router: replica ID must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if st, ok := r.replicas[rep.ID]; ok {
		st.Replica = rep
		return nil
	}
	r.replicas[rep.ID] = &replicaState{Replica: rep, healthy: true}
	r.order = append(r.order, rep.ID)
	sort.Strings(r.order)
	return nil
}

// Remove deregisters a replica and drops every binding pointing at it, so that
// a scale-down does not leave requests chasing a replica that no longer exists.
func (r *Router) Remove(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.replicas, id)
	for i, v := range r.order {
		if v == id {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
	for k, b := range r.affinity {
		if b.replica == id {
			delete(r.affinity, k)
		}
	}
}

// SetHealthy marks a replica in or out of rotation.
func (r *Router) SetHealthy(id string, healthy bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	st, ok := r.replicas[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrUnknownReplica, id)
	}
	st.healthy = healthy
	return nil
}

// Done releases a slot on a replica and records the outcome. Every successful
// Pick must be matched by exactly one Done, so callers should defer it.
type Done func(cacheHit bool)

// Pick chooses a replica for a request with the given prefix key.
func (r *Router) Pick(prefixKey string) (Decision, Done, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.replicas) == 0 {
		return Decision{}, nil, ErrNoReplicas
	}

	now := r.cfg.Now()
	d := r.decideLocked(prefixKey, now)
	if d.Replica == "" {
		return Decision{}, nil, ErrAllUnhealthy
	}

	st := r.replicas[d.Replica]
	st.inFlight++
	st.served++
	d.Load = st.inFlight
	r.decisions[d.Reason]++

	if prefixKey != "" {
		r.bindLocked(prefixKey, d.Replica, now)
	}

	id := d.Replica
	var once sync.Once
	return d, func(cacheHit bool) {
		once.Do(func() {
			r.mu.Lock()
			defer r.mu.Unlock()
			if st, ok := r.replicas[id]; ok {
				if st.inFlight > 0 {
					st.inFlight--
				}
				if cacheHit {
					st.cacheHits++
				}
			}
		})
	}, nil
}

func (r *Router) decideLocked(prefixKey string, now time.Time) Decision {
	least, leastLoad := r.leastLoadedLocked()

	if prefixKey == "" {
		if least == "" {
			return Decision{}
		}
		return Decision{Replica: least, Reason: ReasonLeastLoaded}
	}

	if b, ok := r.affinity[prefixKey]; ok && now.Sub(b.touched) <= r.cfg.AffinityTTL {
		st, exists := r.replicas[b.replica]
		switch {
		case !exists || !st.healthy:
			if least == "" {
				return Decision{}
			}
			return Decision{Replica: least, Reason: ReasonFailover, AffineTo: b.replica}
		case !st.available():
			if least == "" {
				return Decision{}
			}
			return Decision{Replica: least, Reason: ReasonDefect, AffineTo: b.replica}
		case st.load() <= leastLoad*r.cfg.AffinityFactor+1e-9:
			// Close enough to the least loaded replica that keeping the warm
			// cache is worth more than the queueing difference.
			return Decision{Replica: b.replica, Reason: ReasonAffinity, AffineTo: b.replica}
		default:
			if least == "" {
				return Decision{}
			}
			return Decision{Replica: least, Reason: ReasonDefect, AffineTo: b.replica}
		}
	}

	// Unseen prefix: rendezvous hashing sends it somewhere deterministic so a
	// repeat lands on the same replica, without any shared state between
	// router instances.
	if id := r.rendezvousLocked(prefixKey); id != "" {
		return Decision{Replica: id, Reason: ReasonRendezvous}
	}
	if least == "" {
		return Decision{}
	}
	return Decision{Replica: least, Reason: ReasonLeastLoaded}
}

func (r *Router) leastLoadedLocked() (string, float64) {
	best := ""
	bestLoad := 0.0
	for _, id := range r.order {
		st := r.replicas[id]
		if !st.available() {
			continue
		}
		l := st.load()
		if best == "" || l < bestLoad {
			best, bestLoad = id, l
		}
	}
	return best, bestLoad
}

// rendezvousLocked implements highest-random-weight hashing. Unlike a hash ring
// it needs no virtual nodes to distribute evenly, and removing a replica only
// remaps the keys that were on it.
func (r *Router) rendezvousLocked(key string) string {
	best := ""
	var bestScore uint64
	for _, id := range r.order {
		st := r.replicas[id]
		if !st.available() {
			continue
		}
		h := fnv.New64a()
		h.Write([]byte(key))
		h.Write([]byte{0})
		h.Write([]byte(id))
		s := h.Sum64()
		if w := st.Weight; w > 0 && w != 1 {
			s = uint64(float64(s) * w)
		}
		if best == "" || s > bestScore {
			best, bestScore = id, s
		}
	}
	return best
}

func (r *Router) bindLocked(key, replica string, now time.Time) {
	if b, ok := r.affinity[key]; ok {
		b.replica = replica
		b.touched = now
		return
	}
	if len(r.affinity) >= r.cfg.MaxAffinityEntries {
		r.evictLocked(now)
	}
	r.affinity[key] = &binding{replica: replica, touched: now}
}

// evictLocked drops expired bindings, and if that frees nothing, drops the
// oldest. Sampling rather than sorting keeps this O(1)-ish on a large table.
func (r *Router) evictLocked(now time.Time) {
	oldestKey := ""
	var oldest time.Time
	scanned := 0
	for k, b := range r.affinity {
		if now.Sub(b.touched) > r.cfg.AffinityTTL {
			delete(r.affinity, k)
			return
		}
		if oldestKey == "" || b.touched.Before(oldest) {
			oldestKey, oldest = k, b.touched
		}
		scanned++
		if scanned >= 64 {
			break
		}
	}
	if oldestKey != "" {
		delete(r.affinity, oldestKey)
	}
}

// ReplicaState is a snapshot of one replica.
type ReplicaState struct {
	ID        string
	InFlight  int
	Healthy   bool
	Served    uint64
	CacheHits uint64
	Load      float64
}

// Snapshot reports fleet state, sorted by replica ID.
func (r *Router) Snapshot() []ReplicaState {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]ReplicaState, 0, len(r.order))
	for _, id := range r.order {
		st := r.replicas[id]
		out = append(out, ReplicaState{
			ID:        st.ID,
			InFlight:  st.inFlight,
			Healthy:   st.healthy,
			Served:    st.served,
			CacheHits: st.cacheHits,
			Load:      st.load(),
		})
	}
	return out
}

// Decisions reports how many times each reason was used, which is the fastest
// way to tell whether an affinity policy is doing anything at all.
func (r *Router) Decisions() map[Reason]uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[Reason]uint64, len(r.decisions))
	for k, v := range r.decisions {
		out[k] = v
	}
	return out
}

// Bindings reports the number of live prefix bindings.
func (r *Router) Bindings() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.affinity)
}
