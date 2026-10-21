# ADR 0001: Put the boundary between the control plane and the data plane, not inside Go

- **Status:** Accepted
- **Date:** 2026-09-13
- **Context owner:** platform engineering

## Context

Go is the language of the infrastructure layer around machine learning — Kubernetes, the NVIDIA GPU
Operator, device plugins, DRA, Karpenter, KServe, every inference gateway in the ecosystem. It is
absent from the layer those systems schedule. A team standardised on Go therefore faces a recurring
question whenever it builds anything GPU-adjacent: how much of the work should be in Go?

Three answers are commonly proposed.

1. **Go all the way down.** Build tensor types, autodiff and kernels in Go, so the whole system is
   one language, one toolchain, one static binary.
2. **Go nowhere near it.** Write the whole serving stack in Python or C++, on the grounds that the
   ecosystem is there.
3. **Draw a boundary.** Keep Go above some line, delegate below it.

This repository measured the cost of option 1 rather than arguing about it. The measurements are in
[docs/FINDINGS.md](../FINDINGS.md); the summary is that the residue after every available mitigation
is still large:

- The cgo boundary costs ~64 ns per call and nothing removes it.
- Batching the boundary buys 1.15× on its own, because `runtime.Pinner` costs more than the crossing
  it amortises; only combining it with off-heap memory reaches 13.9×.
- Each concurrent blocking C call costs an OS thread.
- The gc compiler runs a float32 dot product at 6.6 GB/s where hand-written assembly reaches 88 GB/s.
- Per-request allocation degrades p99 by 8.8× while barely touching the median.
- No Go compiler targets a GPU, so the kernel is Metal Shading Language or CUDA regardless.

Option 2 is also unattractive, and for reasons that are not about performance. The layer above the
model — admission, batching, routing, placement, health, observability, graceful shutdown — is a
concurrency and lifecycle problem, and it is a problem Go solves better than the alternatives. Giving
it up to stay in one language means writing a scheduler in a language whose concurrency story is
weaker, in exchange for kernels almost nobody on the team will ever modify.

## Decision

**Draw the boundary at the model invocation. Everything above it is Go. Everything below it is a
runtime built for tensors, reached over a network hop.**

Concretely:

- The control plane owns admission control, continuous batching, KV-cache-aware routing, fleet
  topology and placement, health and readiness, metrics, and graceful shutdown. It is Go, with no
  cgo and no accelerator dependency, so it builds and tests anywhere.
- The data plane is an implementation of one interface, `backend.Backend`, whose only method that
  matters takes a batch of requests and returns a batch of responses. vLLM, TensorRT-LLM, SGLang and
  a simulator are all just implementations.
- The interface is batch-shaped, not request-shaped. A request-shaped interface would silently
  forbid the most important optimisation in the system.
- Where Go must touch an accelerator directly — a device plugin, a topology probe, a health check —
  it goes through `pkg/device`: explicit allocation, explicit transfers, asynchronous enqueue, one
  synchronisation point. Never a synchronous call per unit of work from many goroutines.
- Buffers that cross into C come from `pkg/memory.Arena`, not from the Go heap, so no pinning is
  required and the collector never sees them.

## Consequences

**Accepted costs.**

- A network hop per batch, on the order of tens of microseconds, against a forward pass of tens of
  milliseconds. Amortised over a batch it is under 0.1% of the request.
- Two processes to deploy, two things to monitor, two failure modes to reason about, and a
  serialisation format to keep compatible.
- The team does not control the kernels. A performance problem inside the model is escalated to a
  different project rather than fixed locally.

**Gains.**

- The control plane is testable end to end on a laptop with no GPU. This repository's entire test
  suite runs in about eight seconds on two vCPUs, which is the difference between a system people
  refactor confidently and one they do not touch.
- The data plane can be replaced without touching the control plane. Inference runtimes are the
  fastest-moving software in the industry; coupling to one is a liability with a short half-life.
- Separate crash domains. A CUDA out-of-memory in the runtime does not take the router, the queue
  or the metrics endpoint with it.
- Independent scaling and independent release cadence.
- The scheduling policies — where the real engineering leverage is — live in ordinary Go, where they
  can be unit tested against a simulator with a modelled launch overhead and a modelled KV cache.

**Rejected alternatives.**

- *In-process runtime via cgo.* Removes the network hop and reintroduces every finding in
  FINDINGS.md, plus a shared crash domain, plus a CUDA toolkit in the build.
- *A Go tensor library.* Findings 4 and 6. Even granting a heroic assembly effort, the ecosystem gap
  is structural: model formats, quantisation tooling, attention variants and compiler stacks all
  assume Python over C++, and they move faster than any single project can track.
- *Python control plane.* Concedes the layer where Go is strongest, in exchange for kernels the team
  will not be modifying anyway.

## Revisit if

- A vendor-neutral compiler IR makes GPU kernels reachable without a C++ toolchain, and a Go frontend
  for it exists and is maintained. WebGPU compute maturing to production quality, or an MLIR-adjacent
  path with real Go bindings, would both count.
- `GOEXPERIMENT=simd` graduates and extends beyond CPU intrinsics.
- cgo call overhead drops by an order of magnitude. Unlikely; it is inherent to the runtime's design,
  not an unoptimised path.

None of these appear imminent. Until one does, this boundary is where it belongs.
