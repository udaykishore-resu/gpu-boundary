# Findings

Every number here comes from `go run ./cmd/boundary-bench` on the reference host below. Re-run it on
your own hardware before quoting any of it; the ratios travel, the absolute microseconds do not.

**Reference host:** linux/amd64, 2 vCPU, Go 1.25.1, cgo enabled, gcc with `-O3 -march=native`.

> **Status:** these are single runs on a 2-vCPU build host, recorded before the repeated-run
> harness existed. Regenerate them as described in [REPRODUCING.md](REPRODUCING.md) before quoting
> any of them. Section 2 now also has a real-GPU counterpart, `cmd/cuda-launch-bench`, and section 7's
> routing cost is measured by `cmd/router-bench` against a Python baseline.

---

## 1. The cgo boundary costs about 64 ns, and nothing removes it

| measurement | value |
|---|---|
| empty Go function call (`//go:noinline`) | 2.80 ns |
| empty C function call through cgo | 66.80 ns |
| C function taking two int32 and returning one | 70.10 ns |
| **boundary tax** | **64.00 ns/call** |
| ratio | 23.9× |

**Interpretation.** This is the floor. Every `cudaMalloc`, every `cublasSaxpy`, every
`MTLComputeCommandEncoder` call pays it before the driver has done anything. It is not a Go defect
exactly — it is the price of the runtime having its own stacks, its own scheduler and a moving
collector, all of which have to be reconciled with the C ABI on every crossing.

**Why it matters for inference.** A transformer decode step issues on the order of
`layers × ops_per_layer` kernel launches. At 30–100 launches per layer and 32 layers, a single token
is 1000–3000 launches. 64 ns × 2000 = 128 µs of pure boundary per token, against a decode step that
might be 10–20 ms on a large model and well under 1 ms on a small one. On the small model it is
material; on the large one it is noise. The right response is to know which regime you are in, which
is exactly what this benchmark is for.

---

## 2. Batching the boundary is worthless unless you also remove pinning

| batch | pinned Go memory | off-heap arena |
|---|---|---|
| 1 | 249.0 ns/launch | 98.0 ns/launch |
| 4 | 271.1 ns/launch (0.92×) | 36.8 ns/launch (2.66×) |
| 16 | 232.7 ns/launch (1.07×) | 22.8 ns/launch (4.30×) |
| 64 | 213.3 ns/launch (1.17×) | 18.4 ns/launch (5.32×) |
| 256 | 216.3 ns/launch (1.15×) | 17.9 ns/launch (5.46×) |

**Pinning cost: 151 ns per descriptor.**

**Interpretation.** This was the surprise. The textbook mitigation for a per-call boundary cost is to
make fewer, bigger calls — build a command buffer, submit once. Implemented naively in Go it buys
15%, because writing a Go pointer into a C structure requires `runtime.Pinner.Pin`, and pinning two
buffers costs more than the crossing being amortised away.

The fix is to stop involving the collector. `memory.Arena` hands out slices backed by an anonymous
`mmap`: the collector never scans the region, the addresses never move, and pointers into it can be
stored in C memory indefinitely with no pinning at all. Combining the two mitigations takes a launch
from 249 ns to 17.9 ns, **13.9× end to end**, where either one alone gets 1.15× or 2.5×.

**Generalisation.** Any Go program handing buffers to a C library repeatedly — not just GPU code:
image codecs, compression, crypto accelerators, DPDK — has this shape. Allocate the buffers off-heap
once, reuse them, and the FFI boundary stops being the bottleneck.

---

## 3. A blocking C call costs an OS thread

| blockers | threads created | observer p99 lateness |
|---|---|---|
| 8 goroutines blocking in Go | 0 | 301 µs |
| 8 goroutines blocking in C | 8 | 309 µs |

**Interpretation.** When a goroutine blocks in Go, the runtime parks it and reuses the thread. When
it blocks inside a C call, the runtime cannot see inside, cannot preempt it, and after ~20 µs hands
the P to a freshly created thread so the other goroutines keep running. Thread count therefore grows
linearly with the number of concurrent synchronous C calls.

On this 2-vCPU host the *latency* difference is inside the noise; the *thread growth* is exact and
reproducible. On a larger host serving hundreds of concurrent requests through a synchronous GPU
API, thread growth in the hundreds is what shows up first — as memory, as scheduler pressure, and
eventually as `runtime: program exceeds 10000-thread limit`.

**Mitigation.** Do not call synchronously from many goroutines. `device.Device` enqueues from any
goroutine and submits from one place, so concurrency lives in Go and the C boundary sees a single
serialised submitter. This is the same reason CUDA streams and Metal command queues exist.

---

## 4. The Go compiler leaves an order of magnitude on the table

| n | implementation | ns/op | GB/s | vs scalar |
|---|---|---|---|---|
| 4096 | scalar Go | 4973 | 6.6 | 1.0× |
| 4096 | 4-accumulator Go | 1597 | 20.5 | 3.1× |
| 4096 | C via cgo, `-O3 -march=native` | 1928 | 17.0 | 2.6× |
| 4096 | hand-written AVX2+FMA assembly | 372 | 88.1 | **13.4×** |

The same measurement on arm64 (aarch64 Linux, Go 1.25.1, `CGO_ENABLED=0`):

| n | implementation | ns/op | GB/s | vs scalar |
|---|---|---|---|---|
| 4096 | scalar Go | 3171 | 10.3 | 1.0× |
| 4096 | 4-accumulator Go | 861 | 38.1 | 3.7× |
| 4096 | hand-written NEON assembly | 451 | 72.6 | **7.0×** |

The arm64 ratio is smaller than amd64's, and for an interesting reason: NEON is 4-wide where AVX2 is
8-wide, and the Go assembler exposes no vector float add and no pairwise float add, so the horizontal
reduction has to move all eight accumulator lanes out through a general purpose register one at a
time. Half the vector width and a clumsier reduction is most of the gap.

**Interpretation, three parts.**

*The gc compiler does not vectorise this loop.* A float32 dot product is the single most
auto-vectorisable pattern in numeric computing, and the scalar Go version runs at 6.6 GB/s on
hardware capable of ten times that. The gc backend optimises for compile speed and predictability,
which is a defensible trade, and this is what it costs.

*Portable Go can claw back 3×* by breaking the FMA dependency chain with four accumulators. That is
free, portable, and the right first move — but it is still 4× short.

*The C compiler beats scalar Go by 2.6× and still loses to the assembly by 5×,* because the C is
compiled without `-ffast-math` and so must respect float associativity in the reduction, just as the
Go version does. This is a fair comparison rather than a flattering one.

*The fast column is not Go.* It is 60 lines of `.s` with its own CPUID feature detection, an XGETBV
check that the OS actually preserves YMM registers across context switches, and a scalar tail. It is
correct, it is fast, and it is exactly the thing Go was designed so that you would not have to write.

Go 1.26's `simd/archsimd` changes this story: the same algorithm becomes readable Go behind
`GOEXPERIMENT=simd`. It is implemented in `pkg/kernel/simdexp_amd64.go` and is the one file this
repository's CI cannot compile.

---

## 5. The garbage collector taxes the tail, not the median

| strategy | p50 | p99 | p99.9 | allocated | collections |
|---|---|---|---|---|---|
| heap (`make` per iteration) | 16.9 µs | 270.8 µs | 482.3 µs | 1875 MB | 610 |
| `sync.Pool` by size class | 10.5 µs | 30.8 µs | 61.2 µs | 1.6 MB | 0 |
| off-heap bump arena | 10.4 µs | 30.1 µs | 59.1 µs | 0.2 MB | 0 |

**Interpretation.** The median improves by 1.6×. The p99 improves by **8.8×** and the p99.9 by 8×.
That asymmetry is the whole finding, and it is why allocation-heavy serving code passes review: a
throughput benchmark shows almost nothing wrong.

Inference SLOs are written against p99 time-to-first-token. A control plane that allocates per
request is buying itself a p99 problem that will be diagnosed three times as a networking issue
before someone reads a heap profile.

**Choosing between the two mitigations.** The pool needs no capacity planning and no lifetime
discipline; use it by default. The arena is for buffers that cross into C, because its memory is
invisible to the collector and needs no pinning — see finding 2, where that property is worth 13.9×.

---

## 6. Go cannot express a GPU kernel, and no mitigation changes that

`pkg/device` presents one interface over three backends. The CPU backend is Go. The Metal backend
is an Objective-C shim compiling Metal Shading Language at startup. The CUDA backend is the CUDA
runtime plus cuBLAS. In both accelerator cases the actual arithmetic is written in another language
and reached through cgo, because there is no Go compiler targeting a GPU and there is no credible
path to one.

The interesting result is structural rather than numeric: **two unrelated vendor APIs collapse to the
same Go shape.** Allocate opaque handles, copy across an explicit boundary, enqueue asynchronously,
synchronise once. Both backends are about 200 lines of Go over about 150 lines of C or Objective-C,
and the `pkg/device` test suite runs unmodified against either.

That is the argument for where to put the abstraction. Not "make Go a tensor language" — that fails
on findings 1 through 5. Instead: make the boundary explicit, keep it thin, keep it asynchronous, and
put everything above it in Go.

---

## 7. The control plane, where none of this applies

From `cmd/boundaryd -replicas=3 -load=150`, against simulated replicas with a 3 ms launch overhead
and a 32-slot prefix cache:

```
admitted=1055  shed=0  rejected=0  prefix_hit_rate=78%
routing: affinity=557  defect-overloaded=436  rendezvous=62
```

**Interpretation.** 78% of requests landed on a replica that already held their prompt prefix, which
on a real runtime is most of the prefill work skipped. The 436 defections are the policy working as
designed: affinity was abandoned whenever the affine replica's load exceeded the least loaded
replica by more than the configured factor, which is what stops a hot conversation from queueing
behind itself.

None of the six limitations above appear anywhere in this layer. There is no cgo, no tensor, no
allocation in the hot path that matters, and the concurrency — thousands of in-flight requests, each
cancellable, each with a deadline — is the exact workload Go's runtime was built for. The routing
decision for a request takes a few hundred nanoseconds and the forward pass takes tens of
milliseconds, so the control plane is four to five orders of magnitude away from being the
bottleneck.

**The conclusion the numbers support:** Go loses the data plane permanently and owns the control
plane comfortably. Both halves of that sentence are worth acting on.
