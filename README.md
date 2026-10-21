# gpu-boundary

**Where Go's data plane ends and its control plane begins — measured, not asserted.**

Go is the language of Kubernetes, of the NVIDIA GPU Operator, of Karpenter, KServe and every
inference gateway worth naming. It is also nowhere near the tensors those systems schedule. This
repository takes that observation seriously enough to measure it: every limitation people cite
when they say "Go can't do AI" is reproduced here as a runnable benchmark, each one is paired with
the mitigation that actually works, and the residue — the part no mitigation closes — is used to
draw the boundary and build the thing Go *is* the right language for.

Zero dependencies. Standard library only. `go test ./...` passes on a laptop with no GPU in it.

```
go run ./cmd/boundary-bench        # reproduce every measurement below
go run ./cmd/boundaryd -load=200   # run the control plane with synthetic traffic
```

---

## The six limitations, and what this repo does about each

| # | Limitation | Where | Mitigation implemented | Measured residue |
|---|---|---|---|---|
| 1 | Every GPU API is C; Go can only reach one through cgo | `pkg/ffi` | Command-buffer batching: N launches, one crossing | ~64 ns per crossing, irreducible |
| 2 | A blocking C call cannot be preempted, so the runtime spawns threads | `pkg/ffi` | Measurement + async submission model | +8 OS threads for 8 concurrent blocking calls, vs +0 in Go |
| 3 | Go pointers cannot be stored in C memory without pinning | `pkg/memory`, `pkg/ffi` | Off-heap arena; device buffers the collector never sees | Pinning costs ~151 ns/descriptor — more than the crossing it batches away |
| 4 | The gc compiler does not vectorise numeric code | `pkg/kernel` | Hand-written AVX2 and NEON assembly behind runtime dispatch | 11× over scalar Go, but it is assembly, not Go |
| 5 | No arenas, no alignment control, GC in the hot path | `pkg/memory`, `pkg/gcprobe` | Bump-allocated mmap arena + size-classed pool | p99 latency 9× better; throughput barely moves — the cost was always in the tail |
| 6 | Go cannot express a GPU kernel at all | `pkg/device` | One `Device` interface over CPU, Metal and CUDA backends | The kernel is Metal Shading Language or cuBLAS. There is no Go version. |

Nothing in that table is a workaround that makes Go competitive at training or at kernel authoring.
Read the rightmost column: the residue is real, and it is the argument for the second half of the
repository.

## The other half: the control plane

`controlplane/` is a working inference gateway, and it is where the same language stops fighting
and starts winning:

- **`admission/`** — concurrency ceiling, bounded queue, and deadline-aware shedding. Refusing work
  that cannot finish in time is the difference between a system that degrades and one that collapses.
- **`router/`** — KV-cache-aware placement. Prefix affinity with a defection threshold, plus
  rendezvous hashing so independent gateway instances agree on where an unseen prefix goes.
  Measured 78% prefix cache hit rate under Zipf-distributed session traffic.
- **`batcher/`** — generic continuous batching with per-caller cancellation. One caller giving up
  never takes the batch down with it.
- **`topology/`** — device inventory and interconnect-aware placement. A four-way tensor-parallel
  replica is never split across NVLink islands, because a layout that is valid and slow is the
  worst kind of bug.
- **`telemetry/`** — Prometheus exposition, written from scratch to keep the dependency count at zero.
- **`server.go`** — admission → routing → batching, in that order, with the reasoning for the order
  in the source.

The backend is an interface. `backend.Sim` models launch overhead and a KV cache so the whole thing
is testable without hardware; swapping in vLLM or TensorRT-LLM over HTTP is one implementation of
one interface.

---

## Measurements

Taken on linux/amd64, 2 vCPU, Go 1.25.1, cgo enabled. Your numbers will differ; that is the point of
shipping the harness rather than a screenshot.

For numbers you intend to cite, use the repeated-run harness described in
[docs/REPRODUCING.md](docs/REPRODUCING.md). `make bench-paper` runs every measurement five times in
fresh processes and writes the raw data and summary statistics to `results/`.

### The cgo boundary

```
go function call         2.80 ns/op    //go:noinline empty Go function
cgo call                66.80 ns/op    empty C function
boundary tax            64.00 ns/call  paid before any GPU API does anything
tax ratio                23.9 x
```

A transformer forward pass issues thousands of kernel launches. At 64 ns of pure overhead each,
the boundary alone is a material fraction of a small-batch decode step.

### Batching, and why it does not work on its own

```
batch=1   heap    249.0 ns/launch
batch=256 heap    216.3 ns/launch   1.15x vs unbatched   <-- batching alone: almost nothing
batch=1   arena    98.0 ns/launch
batch=256 arena    17.9 ns/launch   5.46x vs unbatched   <-- batching + off-heap: 13.9x end to end
pinning cost      151.0 ns/descriptor
```

This is the most useful finding in the repository and it was not the expected one. Batching saves
the crossing, then `runtime.Pinner` spends the saving back, because every Go slice written into a C
descriptor has to be pinned. The two cancel. Feed the queue memory the collector does not manage and
the pinning disappears — which is why `pkg/memory` and `pkg/ffi` are designed to be used together.

### The scheduler

```
threads created, blocking in Go     0    goroutines are parked, threads reused
threads created, blocking in C      8    one new OS thread per concurrent blocking call
```

Any Go service that calls a synchronous GPU API from many goroutines grows its thread count with
concurrency. The fix is not a tuning knob; it is designing for asynchronous submission, which is
what `device.Device` does.

### CPU kernels

```
n=4096  scalar     4973 ns/op    6.6 GB/s
n=4096  unrolled   1597 ns/op   20.5 GB/s   3.1x  (portable Go, 4 accumulators)
n=4096  cgo-c      1928 ns/op   17.0 GB/s   2.6x  (C with -O3 -march=native, through cgo)
n=4096  avx2-asm    372 ns/op   88.1 GB/s  13.4x  (hand-written Go assembly)
```

On arm64 (verified on real aarch64 hardware): scalar 3171 ns/op → NEON assembly 451 ns/op, **7.0×**
at 72.6 GB/s. Smaller ratio, because NEON is 4-wide and the Go assembler exposes no vector float add,
so the horizontal reduction has to leave the vector registers one lane at a time.

Two things worth sitting with. The Go compiler leaves an order of magnitude on the table for a loop
this simple. And the way to get it back is to stop writing Go — the fastest column is assembly, with
its own CPUID feature detection, because the standard library's is in `internal/cpu` and
`golang.org/x/sys` is a dependency this repo declines to take.

Go 1.26's `simd/archsimd` intrinsics are implemented in `pkg/kernel/simdexp_amd64.go` behind
`GOEXPERIMENT=simd`. It is the one file CI cannot compile — see *Honest limitations* below.

### Allocation and the tail

```
strategy   p50        p99        p99.9      allocated   collections
heap       16.9 us    270.8 us   482.3 us   1875 MB     610
pool       10.5 us     30.8 us    61.2 us      1.6 MB     0
arena      10.4 us     30.1 us    59.1 us      0.2 MB     0
```

Throughput moves a little. p99 moves 9×. This is why allocation-per-request looks fine in a
benchmark and shows up as an SLO violation in production: the collector does not make the median
slower, it makes the tail wider, and inference SLOs are written against the tail.

---

## Layout

```
pkg/ffi/          cgo boundary cost, command-buffer batching, scheduler stall measurement
pkg/kernel/       scalar / unrolled / AVX2 asm / NEON asm / archsimd dot product + dispatch table
pkg/kernel/cref/  the same kernel in C, for the auto-vectoriser comparison
pkg/memory/       off-heap bump arena (mmap) and size-classed buffer pool
pkg/gcprobe/      latency-distribution probe across the three allocation strategies
pkg/device/       Device interface; CPU, Metal (Objective-C shim) and CUDA (cuBLAS) backends
controlplane/     admission, router, batcher, topology, telemetry, and the server that wires them
cmd/boundary-bench  runs every measurement, table or JSON
cmd/boundaryd       the control plane, with an optional synthetic load generator
cmd/bench-runs      repeats any measurement in fresh processes; writes results/<host>-<time>.json
cmd/cuda-launch-bench  section 2 against a real GPU: launch, batching and pinning costs (-tags cuda)
cmd/router-bench    routing-decision cost, same harness as the Python baseline
bench/python/       line-for-line Python port of the router, for the cross-language comparison
internal/routerload/  deterministic routing workload shared by Go and Python
bench/colab/        one-command runner for a Google Colab GPU runtime
results/            measured results, one JSON file per machine and configuration
internal/gated/     parses the build-tag-excluded files so CI catches typos in GPU code
docs/FINDINGS.md    the numbers, with interpretation
docs/REPRODUCING.md how to produce citable numbers: machines, commands, rules for quoting
docs/adr/           why the data plane / control plane boundary sits where it does
```

## Build matrix

| Command | What it exercises |
|---|---|
| `make test` | everything the default build can reach, with `-race` |
| `make bench` | the full measurement suite |
| `CGO_ENABLED=0 make test` | pure-Go fallbacks; proves cgo is optional everywhere it should be |
| `make test-metal` | Metal backend on macOS (requires Xcode command line tools) |
| `make test-cuda` | CUDA + cuBLAS backend (requires the CUDA toolkit) |
| `make test-simd` | `GOEXPERIMENT=simd` intrinsics path (requires Go 1.26+) |
| `make cross` | builds for linux/darwin/windows × amd64/arm64 |
| `make router-crosscheck` | Go and Python routers make the same 100k decisions |
| `make bench-paper` | tests, then every measurement, 5 runs each, saved to `results/` |

## Honest limitations

Stating these is part of the point; a repository about measuring things should be clear about what
it did not measure.

- **The Metal and CUDA backends are written but not executed by CI.** No macOS runner, no NVIDIA
  GPU. `internal/gated` parses them on every run and checks the Objective-C for balanced braces,
  which catches half-finished edits but is not a compile. If you have the hardware, `make test-metal`
  or `make test-cuda` runs the full `pkg/device` suite against it, unchanged.
- **`simdexp_amd64.go` was written against the published `simd/archsimd` API but not compiled**, because
  the toolchain available to the author was Go 1.25.1. It is behind a build tag and cannot affect
  any other build.
- **The NEON kernel was verified by cross-compiling the test suite and running it on real aarch64
  hardware** (all tests pass; 451 ns/op at n=4096, 7.0× over scalar Go), not by the amd64 CI job.
  `make test` on any arm64 box re-verifies it in seconds.
- **`backend.Sim` is a model, not a GPU.** It reproduces the two behaviours the control plane is
  designed around — fixed launch overhead and a prefix cache — and nothing else. The control plane
  numbers show that the *policies* work against a system with those properties; they are not
  throughput claims about any real model server.
- **The CUDA paths have run on exactly one GPU so far**: a Tesla T4 on Google Colab, driver 580,
  CUDA 13.0, with `make test-cuda` passing and `cmd/cuda-launch-bench` v0.2.0 results in
  [`results/`](results). The v0.2.1 additions (launch-only, CUDA Graph, spread-arena variants) are
  compiled and linked against CUDA 12.9 but not yet run. Every variant reads its result back and
  checks it against the exact expected value, so a broken variant fails rather than reporting a number.
- **The 2-vCPU measurement host is small**, which compresses the scheduler-lateness figures. Thread
  growth and allocation figures are structural and reproduce anywhere; the microsecond latencies are
  not portable.

## Skills demonstrated

Go runtime internals (scheduler behaviour under cgo, `runtime.Pinner`, GC tuning and pause
attribution) · hand-written Go assembly for amd64 (AVX2/FMA, CPUID feature detection, XGETBV OS
support checks) and arm64 (NEON, FMLA) · cgo and FFI design, including command-buffer amortisation
and pointer-provenance rules · off-heap memory management with `mmap`, alignment and lifetime
discipline · GPU programming across two vendor APIs — Metal Shading Language with an Objective-C
shim, CUDA runtime with cuBLAS · inference platform engineering: continuous batching, KV-cache-aware
routing, deadline-based load shedding, interconnect-aware placement · Prometheus instrumentation
from first principles · generics, build-tag portability, race-clean concurrent design, and a test
suite that asserts invariants rather than timings.

## Topics

`golang` `gpu` `cuda` `metal` `cgo` `simd` `assembly` `performance` `inference` `llm-serving`
`kv-cache` `continuous-batching` `load-shedding` `control-plane` `platform-engineering`
`benchmarking` `memory-management` `garbage-collection` `prometheus` `zero-dependencies`

## License

MIT. See [LICENSE](LICENSE).
