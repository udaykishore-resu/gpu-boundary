# Changelog

## v0.2.1

Follow-up to the first real-GPU run (Tesla T4, v0.2.0), which showed that a
CUDA launch costs about 5 µs, that Go adds about 1 µs per launch over C, and
that the pinning effect measured on the CPU stand-in is not visible on the
GPU. This release adds what is needed to explain those numbers.

- `cmd/cuda-launch-bench`
  - `launch-only …` variants: a launch with no host memory, isolating what cgo
    adds to the launch itself.
  - `graph K=N …` variants: N launches captured into a CUDA graph and replayed
    with one `cudaGraphLaunch`, from C and from Go.
  - `batch=N arena-spread`: one host block per page, to test whether block
    adjacency explains the arena being slower than pinned heap memory at
    batch 16 and 64.
  - Variants run in a shuffled, seeded order (`-order`, `-seed`), so drift
    cannot favour later variants. The seed is recorded in the output.
  - `-lock-thread`; `GOMAXPROCS`, `GODEBUG` and thread locking are recorded in
    the host description.
  - The registered (page-locked) variant uses its own arena, so registration
    cannot affect the unregistered variants.
  - The CUDA include path is set in the cgo flags; `CGO_CFLAGS` is no longer
    needed on a standard `/usr/local/cuda` install (also for `pkg/device`).
- `cmd/bench-runs` records the commit, `git describe`, whether the tree was
  clean outside `results/`, and measurement-relevant environment variables,
  and warns when the tree is dirty.
- `make bench-cuda-launch` also runs a matrix: `GODEBUG=cgocheck=0`,
  `GOMAXPROCS=1`, and `-lock-thread`, 5 runs each.
- Default `RUNS` raised from 5 to 10.
- `cpu kernel` rows run for a fixed 200 ms budget instead of a fixed iteration
  count; at n=256 the old count finished in under a millisecond and varied by
  30% or more between runs.
- `bench/colab/run.sh`: the whole Colab run in one command, writing a zip of the
  results for download.

## v0.2.0

- Repeated-run harness (`cmd/bench-runs`), real-GPU launch benchmark
  (`cmd/cuda-launch-bench`), router benchmarks (`cmd/router-bench`) with a
  line-for-line Python baseline (`bench/python`) and a Go/Python decision
  cross-check, `docs/REPRODUCING.md`.
- Results: MacBook Air (Apple M5) and Google Colab (Tesla T4).
