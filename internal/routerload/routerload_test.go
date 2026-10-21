package routerload

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestKeysAreDeterministicAndSkewed(t *testing.T) {
	a, b := Keys(Seed, 20_000), Keys(Seed, 20_000)
	hot := 0
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("key %d differs between runs: %q vs %q", i, a[i], b[i])
		}
		if !strings.HasPrefix(a[i], "p") || len(a[i]) != 5 {
			t.Fatalf("malformed key %q", a[i])
		}
		if a[i] < "p0050" {
			hot++
		}
	}
	// 80% hot by construction; allow sampling slack.
	if frac := float64(hot) / float64(len(a)); frac < 0.77 || frac > 0.83 {
		t.Fatalf("hot fraction %.3f, want about 0.80", frac)
	}
}

const goldenKeys = "p0028,p0023,p0037,p0040,p0017,p0048,p0047,p0035"

// The first keys are pinned so that a change to the generator, which would
// silently break the Python cross-check, fails here first with a clear diff.
func TestKeysGolden(t *testing.T) {
	got := strings.Join(Keys(Seed, 8), ",")
	want := goldenKeys
	if got != want {
		t.Fatalf("first keys changed:\n got  %s\n want %s\nupdate bench/python/router.py in the same change", got, want)
	}
}

func TestTraceIsDeterministicAndExercisesAffinity(t *testing.T) {
	a := Trace(Seed, 50_000, Replicas, Hold)
	b := Trace(Seed, 50_000, Replicas, Hold)
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Fatalf("trace not deterministic:\n%s\n%s", ja, jb)
	}
	if a.Decisions != 50_000 {
		t.Fatalf("decisions = %d", a.Decisions)
	}
	sum := 0
	for _, v := range a.ByReplica {
		sum += v
	}
	if sum != a.Decisions {
		t.Fatalf("per-replica counts sum to %d, want %d", sum, a.Decisions)
	}
	if a.ByReason["affinity"] == 0 || a.ByReason["rendezvous"] == 0 {
		t.Fatalf("workload does not exercise both affinity and placement: %v", a.ByReason)
	}
	if len(a.ByReplica) != Replicas {
		t.Fatalf("only %d of %d replicas received traffic: %v", len(a.ByReplica), Replicas, a.ByReplica)
	}
}

func TestDrainReleasesEverything(t *testing.T) {
	r := NewRouter(4)
	w := NewWindow(r, Hold)
	for _, k := range Keys(Seed, 1000) {
		w.Step(k)
	}
	w.Drain()
	for _, st := range r.Snapshot() {
		if st.InFlight != 0 {
			t.Fatalf("%s still has %d in flight after Drain", st.ID, st.InFlight)
		}
	}
}
