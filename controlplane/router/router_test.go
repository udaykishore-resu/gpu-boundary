package router_test

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/router"
)

func newRouter(t *testing.T, n int, cfg router.Config) *router.Router {
	t.Helper()
	r := router.New(cfg)
	for i := 0; i < n; i++ {
		if err := r.Add(router.Replica{ID: fmt.Sprintf("r%d", i), MaxConcurrent: 100}); err != nil {
			t.Fatal(err)
		}
	}
	return r
}

func TestEmptyRouter(t *testing.T) {
	r := router.New(router.Config{})
	if _, _, err := r.Pick("x"); !errors.Is(err, router.ErrNoReplicas) {
		t.Fatalf("got %v, want ErrNoReplicas", err)
	}
}

// TestAffinityIsSticky is the core claim: the same prefix goes back to the same
// replica, so its KV cache is worth something.
func TestAffinityIsSticky(t *testing.T) {
	r := newRouter(t, 4, router.Config{})

	first, done, err := r.Pick("conversation-42")
	if err != nil {
		t.Fatal(err)
	}
	done(true)

	for i := 0; i < 50; i++ {
		d, done, err := r.Pick("conversation-42")
		if err != nil {
			t.Fatal(err)
		}
		done(true)
		if d.Replica != first.Replica {
			t.Fatalf("iteration %d went to %s, want %s: affinity is not sticky", i, d.Replica, first.Replica)
		}
	}
	if got := r.Decisions()[router.ReasonAffinity]; got == 0 {
		t.Fatal("no decision was attributed to affinity")
	}
}

// TestRendezvousIsStableAcrossRouters checks that two independent router
// instances place an unseen prefix identically. Without this, a fleet of
// gateways would scatter the same prefix across every replica and the cache
// would never warm.
func TestRendezvousIsStableAcrossRouters(t *testing.T) {
	a := newRouter(t, 5, router.Config{})
	b := newRouter(t, 5, router.Config{})

	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("prefix-%d", i)
		da, doneA, err := a.Pick(key)
		if err != nil {
			t.Fatal(err)
		}
		doneA(false)
		db, doneB, err := b.Pick(key)
		if err != nil {
			t.Fatal(err)
		}
		doneB(false)
		if da.Replica != db.Replica {
			t.Fatalf("key %s: router A chose %s, router B chose %s", key, da.Replica, db.Replica)
		}
	}
}

func TestRendezvousSpreadsKeys(t *testing.T) {
	r := newRouter(t, 4, router.Config{})
	counts := map[string]int{}
	for i := 0; i < 400; i++ {
		d, done, err := r.Pick(fmt.Sprintf("k-%d", i))
		if err != nil {
			t.Fatal(err)
		}
		done(false)
		counts[d.Replica]++
	}
	if len(counts) < 3 {
		t.Fatalf("400 distinct keys landed on only %d replicas: %v", len(counts), counts)
	}
	for id, c := range counts {
		if c == 0 {
			t.Fatalf("replica %s received nothing", id)
		}
	}
	t.Logf("distribution: %v", counts)
}

// TestDefectionUnderLoad is the other half of the policy: affinity must not be
// allowed to queue a hot prefix behind itself forever.
func TestDefectionUnderLoad(t *testing.T) {
	r := newRouter(t, 3, router.Config{AffinityFactor: 1.5})

	// Bind the prefix and hold a pile of requests open on its replica.
	d, _, err := r.Pick("hot")
	if err != nil {
		t.Fatal(err)
	}
	home := d.Replica

	var releases []router.Done
	defected := false
	for i := 0; i < 20; i++ {
		d, done, err := r.Pick("hot")
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, done)
		if d.Replica != home {
			if d.Reason != router.ReasonDefect {
				t.Fatalf("left the affine replica with reason %q, want %q", d.Reason, router.ReasonDefect)
			}
			defected = true
			break
		}
	}
	for _, rel := range releases {
		rel(false)
	}
	if !defected {
		t.Fatal("router never defected from an overloaded affine replica")
	}
}

func TestFailoverWhenAffineReplicaIsUnhealthy(t *testing.T) {
	r := newRouter(t, 3, router.Config{})

	d, done, err := r.Pick("session")
	if err != nil {
		t.Fatal(err)
	}
	done(false)
	home := d.Replica

	if err := r.SetHealthy(home, false); err != nil {
		t.Fatal(err)
	}

	d2, done2, err := r.Pick("session")
	if err != nil {
		t.Fatal(err)
	}
	done2(false)
	if d2.Replica == home {
		t.Fatal("routed to an unhealthy replica")
	}
	if d2.Reason != router.ReasonFailover {
		t.Fatalf("reason = %q, want %q", d2.Reason, router.ReasonFailover)
	}
	if d2.AffineTo != home {
		t.Fatalf("AffineTo = %q, want %q: the decision should still say where it wanted to go", d2.AffineTo, home)
	}
}

func TestAllUnhealthy(t *testing.T) {
	r := newRouter(t, 2, router.Config{})
	r.SetHealthy("r0", false)
	r.SetHealthy("r1", false)
	if _, _, err := r.Pick("x"); !errors.Is(err, router.ErrAllUnhealthy) {
		t.Fatalf("got %v, want ErrAllUnhealthy", err)
	}
}

func TestAffinityExpires(t *testing.T) {
	now := time.Now()
	r := router.New(router.Config{
		AffinityTTL: time.Minute,
		Now:         func() time.Time { return now },
	})
	for i := 0; i < 4; i++ {
		r.Add(router.Replica{ID: fmt.Sprintf("r%d", i)})
	}

	d, done, err := r.Pick("k")
	if err != nil {
		t.Fatal(err)
	}
	done(false)
	home := d.Replica

	// Take the affine replica out of rotation, advance past the TTL, and put
	// it back: the stale binding must no longer force a decision.
	now = now.Add(2 * time.Minute)
	d2, done2, err := r.Pick("k")
	if err != nil {
		t.Fatal(err)
	}
	done2(false)
	if d2.Reason == router.ReasonAffinity {
		t.Fatalf("expired binding to %s was still honoured", home)
	}
}

func TestRemoveDropsBindings(t *testing.T) {
	r := newRouter(t, 3, router.Config{})
	d, done, err := r.Pick("k")
	if err != nil {
		t.Fatal(err)
	}
	done(false)
	if r.Bindings() != 1 {
		t.Fatalf("Bindings = %d, want 1", r.Bindings())
	}
	r.Remove(d.Replica)
	if r.Bindings() != 0 {
		t.Fatalf("Bindings = %d after removing the bound replica, want 0", r.Bindings())
	}
	for _, s := range r.Snapshot() {
		if s.ID == d.Replica {
			t.Fatal("removed replica is still in the snapshot")
		}
	}
}

func TestInFlightAccounting(t *testing.T) {
	r := newRouter(t, 2, router.Config{})
	_, done, err := r.Pick("a")
	if err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, s := range r.Snapshot() {
		total += s.InFlight
	}
	if total != 1 {
		t.Fatalf("in-flight total = %d, want 1", total)
	}
	done(true)
	done(true) // idempotent

	total, hits := 0, uint64(0)
	for _, s := range r.Snapshot() {
		total += s.InFlight
		hits += s.CacheHits
	}
	if total != 0 {
		t.Fatalf("in-flight total = %d after Done, want 0", total)
	}
	if hits != 1 {
		t.Fatalf("cache hits = %d, want 1: Done must be idempotent", hits)
	}
}

func TestMaxConcurrentIsRespected(t *testing.T) {
	r := router.New(router.Config{})
	r.Add(router.Replica{ID: "only", MaxConcurrent: 2})

	a, doneA, err := r.Pick("")
	if err != nil {
		t.Fatal(err)
	}
	_ = a
	_, doneB, err := r.Pick("")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := r.Pick(""); !errors.Is(err, router.ErrAllUnhealthy) {
		t.Fatalf("got %v, want ErrAllUnhealthy once the only replica is at capacity", err)
	}
	doneA(false)
	doneB(false)
}

func TestWeightedFleetPrefersTheBiggerReplica(t *testing.T) {
	r := router.New(router.Config{})
	r.Add(router.Replica{ID: "big", Weight: 4})
	r.Add(router.Replica{ID: "small", Weight: 1})

	counts := map[string]int{}
	var holds []router.Done
	for i := 0; i < 10; i++ {
		d, done, err := r.Pick("")
		if err != nil {
			t.Fatal(err)
		}
		holds = append(holds, done)
		counts[d.Replica]++
	}
	for _, h := range holds {
		h(false)
	}
	if counts["big"] <= counts["small"] {
		t.Fatalf("weighting ignored: big=%d small=%d", counts["big"], counts["small"])
	}
	t.Logf("distribution: %v", counts)
}

func BenchmarkPick(b *testing.B) {
	r := router.New(router.Config{})
	for i := 0; i < 16; i++ {
		r.Add(router.Replica{ID: fmt.Sprintf("r%d", i), MaxConcurrent: 1 << 20})
	}
	keys := make([]string, 1024)
	for i := range keys {
		keys[i] = fmt.Sprintf("prefix-%d", i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, done, err := r.Pick(keys[i%len(keys)])
		if err != nil {
			b.Fatal(err)
		}
		done(true)
	}
}
