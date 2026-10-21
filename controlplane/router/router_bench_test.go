package router_test

// Benchmarks for the routing decision itself: the number FINDINGS.md section 7
// asserted ("a few hundred nanoseconds") without measuring.
//
// Each iteration is one Pick plus the Done of the request picked Hold
// iterations earlier, over the shared deterministic workload in
// internal/routerload. That pairing is what a serving loop actually pays per
// request, and it is exactly what bench/python/bench_router.py times on the
// Python side.
//
//	go test -run '^$' -bench . -benchmem ./controlplane/router/

import (
	"fmt"
	"testing"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/router"
	"github.com/udaykishore-resu/gpu-boundary/internal/routerload"
)

const keySpace = 1 << 16

func BenchmarkPickDone(b *testing.B) {
	keys := routerload.Keys(routerload.Seed, keySpace)
	for _, replicas := range []int{4, 8, 16, 64} {
		b.Run(fmt.Sprintf("replicas=%d", replicas), func(b *testing.B) {
			w := routerload.NewWindow(routerload.NewRouter(replicas), routerload.Hold)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w.Step(keys[i&(keySpace-1)])
			}
			b.StopTimer()
			w.Drain()
		})
	}
}

// BenchmarkPickDoneParallel shares one router across GOMAXPROCS goroutines,
// each with its own in-flight window. The router serialises on one mutex, so
// this shows what that lock costs under contention - the number to check
// before claiming a single router instance can front a large fleet.
func BenchmarkPickDoneParallel(b *testing.B) {
	keys := routerload.Keys(routerload.Seed, keySpace)
	r := routerload.NewRouter(routerload.Replicas * 8) // room for every goroutine's window
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		w := routerload.NewWindow(r, routerload.Hold)
		i := 0
		for pb.Next() {
			w.Step(keys[i&(keySpace-1)])
			i++
		}
		w.Drain()
	})
}

// BenchmarkUnseenPrefix forces the rendezvous path on every request, which is
// the most expensive decision: one hash per replica.
func BenchmarkUnseenPrefix(b *testing.B) {
	keys := make([]string, keySpace)
	for i := range keys {
		keys[i] = fmt.Sprintf("unseen-%06d", i)
	}
	for _, replicas := range []int{8, 64} {
		b.Run(fmt.Sprintf("replicas=%d", replicas), func(b *testing.B) {
			r := router.New(router.Config{MaxAffinityEntries: keySpace})
			for i := 0; i < replicas; i++ {
				_ = r.Add(router.Replica{ID: routerload.ReplicaID(i), MaxConcurrent: 1 << 20, Weight: 1})
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, done, err := r.Pick(keys[i&(keySpace-1)])
				if err != nil {
					b.Fatal(err)
				}
				done(false)
			}
		})
	}
}
