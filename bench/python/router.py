"""A line-for-line Python port of controlplane/router and internal/routerload.

This is the baseline for the paper's control-plane claim. It is written the way
a competent Python engineer would write it for production - plain classes with
__slots__, a lock around shared state, no clever tricks - because the point is
to compare languages, not to compare a tuned Go router with a careless Python
one.

Every function names the Go function it mirrors. If the Go router changes, this
file must change in the same commit; `make router-crosscheck` fails otherwise.

Known, deliberate differences:

* Weighted rendezvous scaling (Go: uint64(float64(s) * w)) overflows to
  implementation-defined values in Go when the product exceeds 2**64. The
  workload uses weight 1 everywhere, so the path is never taken.
* Binding eviction samples the table in map-iteration order, which is random in
  Go and insertion order in Python. The workload's 1000 prefixes never reach
  MaxAffinityEntries, so eviction never runs.
"""

import threading

# --- internal/routerload constants ------------------------------------------

SEED = 0x9E3779B97F4A7C15
PREFIXES = 1000
HOT_PREFIXES = 50
HOT_PERCENT = 80
HOLD = 32
REPLICAS = 8
MAX_CONCURRENT = 64

MASK64 = (1 << 64) - 1

# --- router reasons ------------------------------------------------------------

AFFINITY = "affinity"
DEFECT = "defect-overloaded"
RENDEZVOUS = "rendezvous"
LEAST_LOADED = "least-loaded"
FAILOVER = "failover"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data, h=FNV_OFFSET):
    """hash/fnv New64a; pass h to continue hashing."""
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


class LCG:
    """routerload.lcg (Knuth MMIX)."""

    __slots__ = ("s",)

    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.s


def keys(seed, n):
    """routerload.Keys."""
    g = LCG(seed)
    out = []
    for _ in range(n):
        r = (g.next() >> 33) % 100
        v = g.next() >> 33
        if r < HOT_PERCENT:
            idx = v % HOT_PREFIXES
        else:
            idx = HOT_PREFIXES + v % (PREFIXES - HOT_PREFIXES)
        out.append("p%04d" % idx)
    return out


def replica_id(i):
    """routerload.ReplicaID."""
    return "r%02d" % i


class _Replica:
    """router.replicaState."""

    __slots__ = ("id", "max_concurrent", "weight", "in_flight", "healthy",
                 "served", "cache_hits")

    def __init__(self, rid, max_concurrent, weight):
        self.id = rid
        self.max_concurrent = max_concurrent
        self.weight = weight
        self.in_flight = 0
        self.healthy = True
        self.served = 0
        self.cache_hits = 0

    def load(self):
        w = self.weight if self.weight > 0 else 1.0
        return self.in_flight / w

    def available(self):
        return self.healthy and (self.max_concurrent <= 0 or self.in_flight < self.max_concurrent)


class Router:
    """router.Router with the default Config and an injectable clock."""

    def __init__(self, affinity_factor=2.0, affinity_ttl_s=120.0,
                 max_affinity_entries=100_000, now=None):
        self.affinity_factor = affinity_factor
        self.affinity_ttl_s = affinity_ttl_s
        self.max_affinity_entries = max_affinity_entries
        self.now = now or (lambda: 0.0)
        self.lock = threading.Lock()
        self.replicas = {}
        self.order = []
        self.affinity = {}  # key -> [replica_id, touched]
        self.decisions = {}

    def add(self, rid, max_concurrent, weight):
        """router.Router.Add."""
        with self.lock:
            if rid in self.replicas:
                st = self.replicas[rid]
                st.max_concurrent, st.weight = max_concurrent, weight
                return
            self.replicas[rid] = _Replica(rid, max_concurrent, weight)
            self.order.append(rid)
            self.order.sort()

    def pick(self, key):
        """router.Router.Pick. Returns (replica, reason, done)."""
        with self.lock:
            if not self.replicas:
                raise RuntimeError("router: no replicas registered")
            now = self.now()
            rid, reason = self._decide(key, now)
            if rid == "":
                raise RuntimeError("router: no healthy replica available")
            st = self.replicas[rid]
            st.in_flight += 1
            st.served += 1
            self.decisions[reason] = self.decisions.get(reason, 0) + 1
            if key != "":
                self._bind(key, rid, now)

        released = [False]

        def done(cache_hit):
            with self.lock:
                if released[0]:
                    return
                released[0] = True
                st2 = self.replicas.get(rid)
                if st2 is not None:
                    if st2.in_flight > 0:
                        st2.in_flight -= 1
                    if cache_hit:
                        st2.cache_hits += 1

        return rid, reason, done

    def _decide(self, key, now):
        """router.Router.decideLocked."""
        least, least_load = self._least_loaded()
        if key == "":
            return (least, LEAST_LOADED) if least else ("", "")

        b = self.affinity.get(key)
        if b is not None and now - b[1] <= self.affinity_ttl_s:
            st = self.replicas.get(b[0])
            if st is None or not st.healthy:
                return (least, FAILOVER) if least else ("", "")
            if not st.available():
                return (least, DEFECT) if least else ("", "")
            if st.load() <= least_load * self.affinity_factor + 1e-9:
                return b[0], AFFINITY
            return (least, DEFECT) if least else ("", "")

        rid = self._rendezvous(key)
        if rid:
            return rid, RENDEZVOUS
        return (least, LEAST_LOADED) if least else ("", "")

    def _least_loaded(self):
        """router.Router.leastLoadedLocked."""
        best, best_load = "", 0.0
        for rid in self.order:
            st = self.replicas[rid]
            if not st.available():
                continue
            l = st.load()
            if best == "" or l < best_load:
                best, best_load = rid, l
        return best, best_load

    def _rendezvous(self, key):
        """router.Router.rendezvousLocked."""
        kb = key.encode() + b"\x00"
        prefix = fnv1a64(kb)
        best, best_score = "", 0
        for rid in self.order:
            st = self.replicas[rid]
            if not st.available():
                continue
            s = fnv1a64(rid.encode(), prefix)
            w = st.weight
            if w > 0 and w != 1:
                s = int(float(s) * w) & MASK64
            if best == "" or s > best_score:
                best, best_score = rid, s
        return best

    def _bind(self, key, rid, now):
        """router.Router.bindLocked."""
        b = self.affinity.get(key)
        if b is not None:
            b[0], b[1] = rid, now
            return
        if len(self.affinity) >= self.max_affinity_entries:
            self._evict(now)
        self.affinity[key] = [rid, now]

    def _evict(self, now):
        """router.Router.evictLocked (sampled; see module docstring)."""
        oldest_key, oldest = None, None
        for scanned, (k, b) in enumerate(self.affinity.items()):
            if now - b[1] > self.affinity_ttl_s:
                del self.affinity[k]
                return
            if oldest_key is None or b[1] < oldest:
                oldest_key, oldest = k, b[1]
            if scanned + 1 >= 64:
                break
        if oldest_key is not None:
            del self.affinity[oldest_key]


def new_router(replicas):
    """routerload.NewRouter: frozen clock, default policy."""
    r = Router(now=lambda: 0.0)
    for i in range(replicas):
        r.add(replica_id(i), MAX_CONCURRENT, 1.0)
    return r


class Window:
    """routerload.Window."""

    __slots__ = ("r", "done", "hits", "i")

    def __init__(self, r, hold):
        self.r = r
        self.done = [None] * hold
        self.hits = [False] * hold
        self.i = 0

    def step(self, key):
        rid, reason, done = self.r.pick(key)
        slot = self.i % len(self.done)
        prev = self.done[slot]
        if prev is not None:
            prev(self.hits[slot])
        self.done[slot] = done
        self.hits[slot] = reason == AFFINITY
        self.i += 1
        return rid, reason

    def drain(self):
        for i, d in enumerate(self.done):
            if d is not None:
                d(self.hits[i])
                self.done[i] = None


def trace(seed, n, replicas, hold):
    """routerload.Trace."""
    r = new_router(replicas)
    w = Window(r, hold)
    h = FNV_OFFSET
    by_reason, by_replica = {}, {}
    for k in keys(seed, n):
        rid, reason = w.step(k)
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_replica[rid] = by_replica.get(rid, 0) + 1
        h = fnv1a64(rid.encode() + b" " + reason.encode() + b"\n", h)
    w.drain()
    return {
        "by_reason": by_reason,
        "by_replica": by_replica,
        "decisions": n,
        "hash": format(h, "x"),
        "hold": hold,
        "keys": n,
        "replicas": replicas,
        "seed": seed,
    }
