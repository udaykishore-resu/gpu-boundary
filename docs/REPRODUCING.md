# Reproducing the measurements

Every number this repository makes a claim with has to survive someone else
running the same command on their own machine. This page is how to produce
numbers that do: repeated runs in fresh processes, the hardware recorded
automatically, and the raw data kept beside the summary.

The figures currently in [FINDINGS.md](FINDINGS.md) predate this harness. They
come from single runs on a 2-vCPU build host. Treat them as a preview, and do
not quote them until they have been regenerated as described below.

## What measures what

| Claim | Command | Output section |
|---|---|---|
| The cgo boundary has a fixed per-call cost | `make bench-runs` | `boundary` |
| Batching crossings only pays once pinning is removed (CPU stand-in) | `make bench-runs` | `launch batching` |
| The same, against a real GPU | `make bench-cuda-launch` | `cuda launch` |
| The collector taxes p99, not the median | `make bench-runs` | `gc` |
| A blocking C call costs an OS thread | `make bench-runs` | `stall` |
| A routing decision is sub-microsecond in Go, and how that compares with Python | `make bench-router` | `router` |
| The Python baseline makes exactly the same decisions as Go | `make router-crosscheck` | trace diff |

`make bench-paper` runs every row this machine can run. It adds the CUDA row
automatically when `nvidia-smi` is present.

## The three machines

Run the suite on each of these. One machine is an anecdote; three show the
effect isn't an artefact of one CPU.

1. **An arm64 laptop**, for example an Apple Silicon Mac. Plug it in, close other
   applications, and turn off Low Power Mode.
2. **An amd64 server.** Use a dedicated or "metal" cloud instance if you can,
   because shared vCPUs add noise from neighbours. Use at least 4 cores.
3. **A single-GPU CUDA instance**, any recent NVIDIA datacenter GPU. Note the
   instance type and the GPU model. The results file records the GPU name and
   driver version; it can't record the instance type.

## Steps, on each machine

```sh
# 0. Toolchain: Go 1.24 or newer, a C compiler, Python 3.8+.
#    macOS:  brew install go    (Xcode command line tools provide clang)
go version && cc --version && python3 --version

# 1. The exact code you are measuring.
git clone https://github.com/udaykishore-resu/gpu-boundary && cd gpu-boundary
git checkout <tag>            # always measure a tagged commit, never a dirty tree

# 2. Correctness first. A benchmark of broken code is worthless.
make test
make router-crosscheck

# 3. Measurements. RUNS defaults to 5. LABEL names the machine in the file name.
make bench-paper LABEL=m3-air          # or LABEL=aws-c7i-metal, LABEL=aws-g6-l4, ...
```

Each run of `bench-runs` writes one file to `results/`, named
`<label>-<UTC time>.json`. The file contains:

- `host`: GOOS, GOARCH, CPU count, Go version, and the GPU name and CUDA driver
  and runtime versions for the CUDA row.
- `machine`: the CPU model, kernel version, and `nvidia-smi` GPU line.
- `command`: the exact command line that produced the numbers.
- `summary`: for every measurement, the median, min, max, mean, sample standard
  deviation, and coefficient of variation across runs.
- `raw`: every run's findings, unmodified.

Commit the `results/*.json` files exactly as written, in the same tagged release
the paper cites. Never edit them by hand.

## Rules for quoting a number

1. **Quote the median, with the min–max range alongside.** Example: "64 ns
   (63–67, n=5)".
2. **Check the cv % column.** If it's above about 5% for something you want to
   quote, the machine was noisy. Find the cause (thermal throttling, background
   load, a shared vCPU) and rerun before trusting the median.
3. **Prefer ratios to absolute values** when comparing across machines. The
   absolute microseconds won't transfer between machines; the ratios usually do.
4. **Name the machine every time**: the CPU or GPU model, and the Go version.

## Notes on specific measurements

**`cuda launch`.** Each "launch" is a host-to-device copy of a small block plus
a cuBLAS saxpy on the same stream. That is the decode-step pattern of "upload
parameters, run kernel".

- **The stream is synchronised every `-sync-every` launches (default 1024)** so
  the driver's queue never fills. Otherwise the benchmark would measure GPU
  throughput instead of launch cost.
- **Every variant's output is read back and checked against the exact expected
  value.** A variant that silently did nothing fails instead of reporting a
  flattering number.
- **The `arena+registered` variant page-locks the arena with `cudaHostRegister`.**
  Some virtualised GPUs refuse that call; the row then reports "skipped" and the
  rest of the table is still valid.
- **Building it needs the CUDA toolkit:** `go build -tags cuda`. It has been
  compiled and linked against CUDA 12.9 headers and libraries, but not yet run
  on a GPU. The first run on real hardware is the test.

**`router`.**

- **Both languages time the identical loop:** one Pick plus one Done per request,
  over the same deterministic workload, with one warmup pass and then the median
  of the timed passes.
- **`make router-crosscheck` comes first.** It confirms that both implementations
  produce the same 100,000 decisions, including a hash of the full sequence.
  Don't publish a Go-versus-Python ratio unless the cross-check has passed on
  that commit.
- **Python version matters.** The output records the interpreter and version, so
  run it under the version a real deployment would use.
- **For contention and allocation figures**, which the cross-language harness
  doesn't cover, use the Go benchmarks:
  `go test -run '^$' -bench . -benchmem ./controlplane/router/`.

**`gc`, `stall`.** These are sensitive to core count. On small machines the
`stall` latency difference falls inside the noise, and the thread growth is the
reproducible part. Say which one you are quoting.

## Before a number goes into the paper

- [ ] Measured on a tagged commit, and the tag is cited
- [ ] `make test` and `make router-crosscheck` passed on that commit
- [ ] At least 5 runs on each machine; cv % checked
- [ ] Results JSON committed unmodified
- [ ] Machine, Go version and, for GPU rows, driver version stated beside the number
