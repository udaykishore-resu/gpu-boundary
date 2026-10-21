//go:build cuda && cgo

// Command cuda-launch-bench measures what FINDINGS.md section 2 measured, but
// against a real GPU instead of a C function standing in for one.
//
// The question is the same: when Go drives a device through cgo, what does each
// launch cost, how much does batching launches into one crossing recover, and
// how much of that recovery does runtime.Pinner give back when the launch
// descriptors reference Go heap memory?
//
// v0.2.0 answered the first half and left two things unexplained: Go costs
// about 1 µs more per launch than C even though a bare cgo call costs ~40 ns,
// and at some batch sizes the off-heap arena was slower than pinned heap
// memory. v0.2.1 adds the variants needed to take those apart:
//
//	upload+launch   (v0.2.0) copy a small host block to the device, then saxpy
//	launch-only     saxpy alone, no host memory involved at all, so neither
//	                pinning nor the copy can contribute
//	graph           K saxpy launches captured into one CUDA graph and replayed
//	                with a single cudaGraphLaunch, issued from C or from Go
//	arena-spread    the arena variant with every host block on its own page,
//	                to test whether block adjacency explains the v0.2.0
//	                arena slowdown
//
// Variants run in a shuffled order (seed recorded as a finding) so that GPU
// clock ramp-up or any other drift cannot favour whichever variant happens to
// run later. -lock-thread pins the measuring goroutine to one OS thread, and
// GOMAXPROCS and GODEBUG are recorded in the host description, so the
// Makefile's matrix can isolate scheduler and cgocheck effects.
//
// Every variant synchronises the stream every `-sync-every` launches so the
// driver's launch queue never fills, and every variant's result buffer is read
// back and checked against the exact expected value, so a variant that
// silently launched nothing cannot report a flattering number.
//
// Build and run on a machine with the CUDA toolkit:
//
//	go build -tags cuda -o bin/cuda-launch-bench ./cmd/cuda-launch-bench
//	bin/cuda-launch-bench               # table
//	bin/cuda-launch-bench -json         # for cmd/bench-runs
package main

/*
#cgo CFLAGS: -O2
#cgo linux CFLAGS: -I/usr/local/cuda/include
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcublas
#cgo windows LDFLAGS: -lcudart -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t  lbBlas   = NULL;
static cudaStream_t    lbStream = NULL;
static float*          lbX      = NULL;
static float*          lbY      = NULL;
static void*           lbWS     = NULL;
static cudaGraphExec_t lbGraph  = NULL;
static int             lbN      = 0;
static const float     lbOne    = 1.0f;
static char            lbErr[512];

#define LB_WORKSPACE_BYTES (4u << 20)

static void lb_set_err(const char* where, const char* what) {
    snprintf(lbErr, sizeof(lbErr), "%s: %s", where, what ? what : "unknown error");
}

static const char* lb_error(void) { return lbErr; }

// lcmd is one launch descriptor: where the host parameter block lives, the
// scalar, and the problem size. `host` is the field that forces pinning when
// it points into the Go heap.
typedef struct {
    const float* host;
    float        alpha;
    int32_t      n;
} lcmd;

static int lb_init(int n, char* name, int nameCap, int* driverVer, int* runtimeVer) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) { lb_set_err("cudaGetDeviceCount", cudaGetErrorString(e)); return 1; }
    if (count == 0)       { lb_set_err("init", "no CUDA devices present"); return 2; }

    struct cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, 0);
    if (e != cudaSuccess) { lb_set_err("cudaGetDeviceProperties", cudaGetErrorString(e)); return 3; }
    snprintf(name, nameCap, "%s (sm_%d%d)", prop.name, prop.major, prop.minor);
    cudaDriverGetVersion(driverVer);
    cudaRuntimeGetVersion(runtimeVer);

    e = cudaStreamCreateWithFlags(&lbStream, cudaStreamNonBlocking);
    if (e != cudaSuccess) { lb_set_err("cudaStreamCreate", cudaGetErrorString(e)); return 4; }
    if (cublasCreate(&lbBlas) != CUBLAS_STATUS_SUCCESS) { lb_set_err("cublasCreate", "failed"); return 5; }
    if (cublasSetStream(lbBlas, lbStream) != CUBLAS_STATUS_SUCCESS) { lb_set_err("cublasSetStream", "failed"); return 6; }

    // A fixed workspace stops cuBLAS allocating device memory while a graph
    // is being captured, which capture does not allow.
    e = cudaMalloc(&lbWS, LB_WORKSPACE_BYTES);
    if (e != cudaSuccess) { lb_set_err("cudaMalloc workspace", cudaGetErrorString(e)); return 7; }
    if (cublasSetWorkspace(lbBlas, lbWS, LB_WORKSPACE_BYTES) != CUBLAS_STATUS_SUCCESS) {
        lb_set_err("cublasSetWorkspace", "failed"); return 8;
    }

    lbN = n;
    e = cudaMalloc((void**)&lbX, (size_t)n * sizeof(float));
    if (e != cudaSuccess) { lb_set_err("cudaMalloc x", cudaGetErrorString(e)); return 9; }
    e = cudaMalloc((void**)&lbY, (size_t)n * sizeof(float));
    if (e != cudaSuccess) { lb_set_err("cudaMalloc y", cudaGetErrorString(e)); return 10; }

    // x starts as all ones, so the launch-only and graph variants, which never
    // upload, still add exactly 1.0 per launch.
    float* ones = (float*)malloc((size_t)n * sizeof(float));
    if (!ones) { lb_set_err("malloc", "out of memory"); return 11; }
    for (int i = 0; i < n; i++) ones[i] = 1.0f;
    e = cudaMemcpy(lbX, ones, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    free(ones);
    if (e != cudaSuccess) { lb_set_err("cudaMemcpy x", cudaGetErrorString(e)); return 12; }
    return 0;
}

// lb_op is one upload+launch: copy the host block, then saxpy over it.
static int lb_op(const float* host, float alpha, int32_t n) {
    cudaError_t e = cudaMemcpyAsync(lbX, host, (size_t)n * sizeof(float),
                                    cudaMemcpyHostToDevice, lbStream);
    if (e != cudaSuccess) { lb_set_err("cudaMemcpyAsync", cudaGetErrorString(e)); return 1; }
    if (cublasSaxpy(lbBlas, n, &alpha, lbX, 1, lbY, 1) != CUBLAS_STATUS_SUCCESS) {
        lb_set_err("cublasSaxpy", "failed"); return 2;
    }
    return 0;
}

// lb_batch issues k upload+launches for the price of one boundary crossing.
static int lb_batch(const lcmd* cs, int32_t k) {
    int32_t i;
    for (i = 0; i < k; i++) {
        int rc = lb_op(cs[i].host, cs[i].alpha, cs[i].n);
        if (rc) return rc;
    }
    return 0;
}

static int lb_sync(void) {
    cudaError_t e = cudaStreamSynchronize(lbStream);
    if (e != cudaSuccess) { lb_set_err("cudaStreamSynchronize", cudaGetErrorString(e)); return 1; }
    return 0;
}

// lb_native is the upload+launch floor: the same work issued from C.
static int lb_native(const float* host, float alpha, int32_t n, int64_t count, int32_t syncEvery) {
    int64_t i;
    for (i = 0; i < count; i++) {
        int rc = lb_op(host, alpha, n);
        if (rc) return rc;
        if (syncEvery > 0 && (i + 1) % syncEvery == 0) {
            rc = lb_sync();
            if (rc) return rc;
        }
    }
    return lb_sync();
}

// lb_launch is one launch with no host memory involved.
static int lb_launch(int32_t n) {
    if (cublasSaxpy(lbBlas, n, &lbOne, lbX, 1, lbY, 1) != CUBLAS_STATUS_SUCCESS) {
        lb_set_err("cublasSaxpy", "failed"); return 2;
    }
    return 0;
}

static int lb_launch_n(int32_t n, int32_t k) {
    int32_t i;
    for (i = 0; i < k; i++) {
        int rc = lb_launch(n);
        if (rc) return rc;
    }
    return 0;
}

static int lb_native_launch(int32_t n, int64_t count, int32_t syncEvery) {
    int64_t i;
    for (i = 0; i < count; i++) {
        int rc = lb_launch(n);
        if (rc) return rc;
        if (syncEvery > 0 && (i + 1) % syncEvery == 0) {
            rc = lb_sync();
            if (rc) return rc;
        }
    }
    return lb_sync();
}

// lb_graph_build captures k launch-only saxpys into an executable graph,
// replacing any previous one. Capture and the launches happen inside this
// one C call, so they are on one OS thread, as thread-local capture requires.
static int lb_graph_build(int32_t n, int32_t k) {
    if (lbGraph) { cudaGraphExecDestroy(lbGraph); lbGraph = NULL; }
    cudaGraph_t g = NULL;
    cudaError_t e = cudaStreamBeginCapture(lbStream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) { lb_set_err("cudaStreamBeginCapture", cudaGetErrorString(e)); return 1; }
    int32_t i;
    for (i = 0; i < k; i++) {
        if (cublasSaxpy(lbBlas, n, &lbOne, lbX, 1, lbY, 1) != CUBLAS_STATUS_SUCCESS) {
            cudaStreamEndCapture(lbStream, &g);
            if (g) cudaGraphDestroy(g);
            lb_set_err("cublasSaxpy during capture", "failed"); return 2;
        }
    }
    e = cudaStreamEndCapture(lbStream, &g);
    if (e != cudaSuccess) { lb_set_err("cudaStreamEndCapture", cudaGetErrorString(e)); return 3; }
    e = cudaGraphInstantiateWithFlags(&lbGraph, g, 0);
    cudaGraphDestroy(g);
    if (e != cudaSuccess) { lb_set_err("cudaGraphInstantiate", cudaGetErrorString(e)); lbGraph = NULL; return 4; }
    return 0;
}

static int lb_graph_launch(void) {
    cudaError_t e = cudaGraphLaunch(lbGraph, lbStream);
    if (e != cudaSuccess) { lb_set_err("cudaGraphLaunch", cudaGetErrorString(e)); return 1; }
    return 0;
}

static int lb_native_graph(int64_t launches, int32_t syncEvery) {
    int64_t i;
    for (i = 0; i < launches; i++) {
        int rc = lb_graph_launch();
        if (rc) return rc;
        if (syncEvery > 0 && (i + 1) % syncEvery == 0) {
            rc = lb_sync();
            if (rc) return rc;
        }
    }
    return lb_sync();
}

static int lb_reset_y(void) {
    cudaError_t e = cudaMemsetAsync(lbY, 0, (size_t)lbN * sizeof(float), lbStream);
    if (e != cudaSuccess) { lb_set_err("cudaMemsetAsync", cudaGetErrorString(e)); return 1; }
    return lb_sync();
}

static int lb_read_y0(float* out) {
    int rc = lb_sync();
    if (rc) return rc;
    cudaError_t e = cudaMemcpy(out, lbY, sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { lb_set_err("cudaMemcpy y", cudaGetErrorString(e)); return 1; }
    return 0;
}

static int lb_register(void* p, size_t bytes) {
    cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
    if (e != cudaSuccess) { lb_set_err("cudaHostRegister", cudaGetErrorString(e)); return 1; }
    return 0;
}

static void lb_unregister(void* p) { if (p) cudaHostUnregister(p); }

static void lb_shutdown(void) {
    if (lbGraph)  { cudaGraphExecDestroy(lbGraph); lbGraph = NULL; }
    if (lbX)      { cudaFree(lbX); lbX = NULL; }
    if (lbY)      { cudaFree(lbY); lbY = NULL; }
    if (lbWS)     { cudaFree(lbWS); lbWS = NULL; }
    if (lbBlas)   { cublasDestroy(lbBlas); lbBlas = NULL; }
    if (lbStream) { cudaStreamDestroy(lbStream); lbStream = NULL; }
}
*/
import "C"

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"text/tabwriter"
	"time"
	"unsafe"

	"github.com/udaykishore-resu/gpu-boundary/pkg/memory"
)

// Finding and report match cmd/boundary-bench, so cmd/bench-runs can
// aggregate either.
type Finding struct {
	Section string  `json:"section"`
	Name    string  `json:"name"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
	Note    string  `json:"note"`
}

type hostInfo struct {
	GOOS         string   `json:"goos"`
	GOARCH       string   `json:"goarch"`
	NumCPU       int      `json:"num_cpu"`
	GOMAXPROCS   int      `json:"gomaxprocs"`
	GODEBUG      string   `json:"godebug"`
	LockedThread bool     `json:"locked_thread"`
	Order        string   `json:"order"`
	GoVersion    string   `json:"go_version"`
	CGO          bool     `json:"cgo"`
	Devices      []string `json:"devices"`
	DeviceName   string   `json:"device_name"`
	CUDADriver   string   `json:"cuda_driver"`
	CUDARuntime  string   `json:"cuda_runtime"`
}

type report struct {
	Host     hostInfo  `json:"host"`
	Findings []Finding `json:"findings"`
}

const section = "cuda launch"

var (
	flagJSON       = flag.Bool("json", false, "emit JSON instead of a table")
	flagQuick      = flag.Bool("quick", false, "reduced iteration count")
	flagIters      = flag.Int("iters", 100_000, "timed launches per variant")
	flagElems      = flag.Int("elems", 16, "floats per host block; small, so the boundary dominates")
	flagSyncEvery  = flag.Int("sync-every", 1024, "synchronise the stream every N launches")
	flagBatches    = flag.String("batches", "1,4,16,64,256", "batch sizes; sizes above 1 are also graph sizes")
	flagOrder      = flag.String("order", "shuffle", "variant order: shuffle or fixed")
	flagSeed       = flag.Int64("seed", -1, "shuffle seed; -1 picks one from the clock and records it")
	flagLockThread = flag.Bool("lock-thread", false, "run every variant on one locked OS thread")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "cuda-launch-bench:", err)
		os.Exit(1)
	}
}

func cudaErr() error { return errors.New(C.GoString(C.lb_error())) }

func check(rc C.int) error {
	if rc != 0 {
		return cudaErr()
	}
	return nil
}

// job is one variant. It issues exactly k launches when called with k, so
// the read-back check can prove the work happened.
type job struct {
	name, note string
	// step is the launch granularity: graph variants can only issue whole
	// graphs, so their counts are rounded to multiples of it.
	step int
	// selfSync marks a job that synchronises on its own schedule.
	selfSync bool
	setup    func() error
	fn       func(k int) error
}

func run() error {
	if *flagLockThread {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	iters := *flagIters
	if *flagQuick {
		iters = 10_000
	}
	n := *flagElems
	if n <= 0 {
		return errors.New("-elems must be positive")
	}
	batches, err := parseBatches(*flagBatches)
	if err != nil {
		return err
	}
	maxBatch := 1
	for _, b := range batches {
		maxBatch = max(maxBatch, b)
	}
	// Every launch adds exactly 1.0 to y[0]; float32 counts exactly up to
	// 2^24, and the correctness check depends on that.
	warm := iters/20 + 1
	if iters+warm+2*maxBatch >= 1<<24 {
		return fmt.Errorf("-iters too large for the exactness check")
	}
	if *flagOrder != "shuffle" && *flagOrder != "fixed" {
		return fmt.Errorf("-order must be shuffle or fixed, got %q", *flagOrder)
	}

	var nameBuf [256]C.char
	var drv, rt C.int
	if err := check(C.lb_init(C.int(n), &nameBuf[0], C.int(len(nameBuf)), &drv, &rt)); err != nil {
		return err
	}
	defer C.lb_shutdown()

	rep := report{Host: hostInfo{
		GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, NumCPU: runtime.NumCPU(),
		GOMAXPROCS: runtime.GOMAXPROCS(0), GODEBUG: os.Getenv("GODEBUG"),
		LockedThread: *flagLockThread, Order: *flagOrder,
		GoVersion: runtime.Version(), CGO: true, Devices: []string{"cuda"},
		DeviceName:  C.GoString(&nameBuf[0]),
		CUDADriver:  cudaVersion(int(drv)),
		CUDARuntime: cudaVersion(int(rt)),
	}}

	page := os.Getpagesize()
	blockBytes := n * 4

	// Host blocks are all ones and alpha is 1, so after k launches y[0] == k.
	heap := make([][]float32, maxBatch)
	for i := range heap {
		heap[i] = ones(make([]float32, n))
	}
	packed, packedBlocks, err := newBlocks(maxBatch, n, memory.DefaultAlign)
	if err != nil {
		return err
	}
	defer packed.Close()
	spread, spreadBlocks, err := newBlocks(maxBatch, n, page)
	if err != nil {
		return err
	}
	defer spread.Close()
	// The registered blocks live in their own arena, so page-locking them
	// cannot change the unregistered variants, whatever order they run in.
	reg, regBlocks, err := newBlocks(maxBatch, n, memory.DefaultAlign)
	if err != nil {
		return err
	}
	defer reg.Close()
	regionPtr, regionBytes := pageSpan(regBlocks, blockBytes)
	regErr := check(C.lb_register(regionPtr, C.size_t(regionBytes)))
	if regErr == nil {
		defer C.lb_unregister(regionPtr)
	}

	m := &measurer{n: n, iters: iters, warm: warm, syncEvery: *flagSyncEvery}
	q, err := newQueue(maxBatch)
	if err != nil {
		return err
	}
	defer q.close()

	// Canonical order: this is also the output order, whatever order the
	// jobs actually run in.
	var jobs []job
	jobs = append(jobs,
		job{name: "native C loop", note: "upload+launch issued from C: the floor", selfSync: true,
			fn: func(k int) error {
				return check(C.lb_native((*C.float)(packedBlocks[0]), 1, C.int32_t(n),
					C.int64_t(k), C.int32_t(m.syncEvery)))
			}},
		job{name: "go unbatched, heap", note: "one cgo call per upload+launch", fn: m.unbatchedHeap(heap[0])},
		job{name: "go unbatched, arena", note: "one cgo call per upload+launch", fn: m.unbatchedPtr(packedBlocks[0])},
	)
	for _, b := range batches {
		jobs = append(jobs,
			job{name: fmt.Sprintf("batch=%-3d heap+pin", b), fn: m.batchedHeap(q, b, heap)},
			job{name: fmt.Sprintf("batch=%-3d arena", b), fn: m.batchedPtr(q, b, packedBlocks)},
			job{name: fmt.Sprintf("batch=%-3d arena-spread", b), note: "one host block per page",
				fn: m.batchedPtr(q, b, spreadBlocks)},
		)
		if regErr == nil {
			jobs = append(jobs, job{name: fmt.Sprintf("batch=%-3d arena+registered", b),
				note: "page-locked with cudaHostRegister", fn: m.batchedPtr(q, b, regBlocks)})
		}
	}
	jobs = append(jobs,
		job{name: "launch-only native C loop", note: "no host memory: launch cost alone", selfSync: true,
			fn: func(k int) error {
				return check(C.lb_native_launch(C.int32_t(n), C.int64_t(k), C.int32_t(m.syncEvery)))
			}},
		job{name: "launch-only go unbatched", note: "one cgo call per launch", fn: m.launchOnly(1)},
	)
	for _, b := range batches {
		if b > 1 {
			jobs = append(jobs, job{name: fmt.Sprintf("launch-only batch=%-3d go", b),
				note: "one cgo call per batch, nothing to pin", fn: m.launchOnly(b)})
		}
	}
	for _, b := range batches {
		if b <= 1 {
			continue
		}
		k := b
		build := func() error { return check(C.lb_graph_build(C.int32_t(n), C.int32_t(k))) }
		jobs = append(jobs,
			job{name: fmt.Sprintf("graph K=%-3d native C", k), note: "per kernel; one cudaGraphLaunch per K kernels",
				step: k, selfSync: true, setup: build,
				fn: func(total int) error {
					return check(C.lb_native_graph(C.int64_t(total/k), C.int32_t(max(1, m.syncEvery/k))))
				}},
			job{name: fmt.Sprintf("graph K=%-3d go", k), note: "per kernel; one cgo call per graph launch",
				step: k, setup: build, fn: m.graph(k)},
		)
	}

	seed := *flagSeed
	if seed < 0 {
		seed = time.Now().UnixNano() % (1 << 31)
	}
	order := make([]int, len(jobs))
	for i := range order {
		order[i] = i
	}
	if *flagOrder == "shuffle" {
		r := rand.New(rand.NewPCG(uint64(seed), 0x9E3779B97F4A7C15))
		r.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	}

	results := make([]float64, len(jobs))
	for _, i := range order {
		v, err := m.run(jobs[i])
		if err != nil {
			return fmt.Errorf("%s: %w", strings.TrimSpace(jobs[i].name), err)
		}
		results[i] = v
	}

	byName := map[string]float64{}
	for i, j := range jobs {
		byName[j.name] = results[i]
	}
	unbHeap := byName["go unbatched, heap"]
	for i, j := range jobs {
		note := j.note
		if strings.HasPrefix(j.name, "batch=") {
			ratio := fmt.Sprintf("%.2fx vs go unbatched heap", safeDiv(unbHeap, results[i]))
			if note == "" {
				note = ratio
			} else {
				note += "; " + ratio
			}
		}
		rep.Findings = append(rep.Findings, Finding{section, j.name, results[i], "ns/launch", note})
	}
	if regErr != nil {
		rep.Findings = append(rep.Findings, Finding{section, "arena+registered", 0, "", "skipped: " + regErr.Error()})
	}

	derived := func(name string, v float64, unit, note string) {
		rep.Findings = append(rep.Findings, Finding{section, name, v, unit, note})
	}
	derived("cgo tax per launch", byName["go unbatched, arena"]-byName["native C loop"], "ns/launch",
		"go unbatched arena minus native C loop (upload+launch)")
	derived("launch-only cgo tax", byName["launch-only go unbatched"]-byName["launch-only native C loop"], "ns/launch",
		"go unbatched minus native C loop, no host memory")
	for _, b := range []int{1, 64} {
		h, okH := byName[fmt.Sprintf("batch=%-3d heap+pin", b)]
		a, okA := byName[fmt.Sprintf("batch=%-3d arena", b)]
		if okH && okA {
			derived(fmt.Sprintf("pinning cost at batch=%d", b), h-a, "ns/descriptor",
				"batched heap+pin minus batched arena")
		}
	}
	for _, b := range batches {
		if b <= 1 {
			continue
		}
		g := byName[fmt.Sprintf("graph K=%-3d go", b)]
		c := byName[fmt.Sprintf("graph K=%-3d native C", b)]
		derived(fmt.Sprintf("graph cgo tax at K=%d", b), g-c, "ns/kernel", "graph go minus graph native C")
	}
	derived("variant order seed", float64(seed), "seed", "order="+*flagOrder+"; rerun with -order fixed or -seed to reproduce")

	if *flagJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}
	printTable(rep)
	return nil
}

// measurer times one job: a short untimed warmup, then iters launches, then a
// read-back that must equal warmup+iters exactly.
type measurer struct {
	n, iters, warm, syncEvery int
}

func roundUp(v, step int) int   { return (v + step - 1) / step * step }
func roundDown(v, step int) int { return max(step, v/step*step) }

func (m *measurer) run(j job) (float64, error) {
	step := max(1, j.step)
	warm, iters := roundUp(m.warm, step), roundDown(m.iters, step)
	if j.setup != nil {
		if err := j.setup(); err != nil {
			return 0, err
		}
	}
	if err := check(C.lb_reset_y()); err != nil {
		return 0, err
	}
	runtime.GC()
	if err := j.fn(warm); err != nil {
		return 0, err
	}
	if !j.selfSync {
		if err := check(C.lb_sync()); err != nil {
			return 0, err
		}
	}
	start := time.Now()
	if err := j.fn(iters); err != nil {
		return 0, err
	}
	if err := check(C.lb_sync()); err != nil {
		return 0, err
	}
	elapsed := time.Since(start)

	var y0 C.float
	if err := check(C.lb_read_y0(&y0)); err != nil {
		return 0, err
	}
	if want := float32(warm + iters); float32(y0) != want {
		return 0, fmt.Errorf("correctness check failed: y[0] = %v, want %v", float32(y0), want)
	}
	return float64(elapsed.Nanoseconds()) / float64(iters), nil
}

func (m *measurer) maybeSync(issued *int, k int) error {
	*issued += k
	if m.syncEvery > 0 && *issued >= m.syncEvery {
		*issued = 0
		return check(C.lb_sync())
	}
	return nil
}

func (m *measurer) unbatchedHeap(block []float32) func(int) error {
	return func(k int) error {
		issued := 0
		for i := 0; i < k; i++ {
			// Passing a Go pointer as a cgo argument is legal without a
			// Pinner: cgo keeps it in place for the duration of the call,
			// and a pageable cudaMemcpyAsync has staged the data before it
			// returns.
			if err := check(C.lb_op((*C.float)(unsafe.Pointer(&block[0])), 1, C.int32_t(m.n))); err != nil {
				return err
			}
			if err := m.maybeSync(&issued, 1); err != nil {
				return err
			}
		}
		return nil
	}
}

func (m *measurer) unbatchedPtr(block unsafe.Pointer) func(int) error {
	return func(k int) error {
		issued := 0
		for i := 0; i < k; i++ {
			if err := check(C.lb_op((*C.float)(block), 1, C.int32_t(m.n))); err != nil {
				return err
			}
			if err := m.maybeSync(&issued, 1); err != nil {
				return err
			}
		}
		return nil
	}
}

func (m *measurer) batchedHeap(q *queue, batch int, blocks [][]float32) func(int) error {
	return func(k int) error {
		issued := 0
		for done := 0; done < k; {
			b := min(batch, k-done)
			for i := 0; i < b; i++ {
				q.addHeap(blocks[i], m.n)
			}
			if err := q.flush(); err != nil {
				return err
			}
			done += b
			if err := m.maybeSync(&issued, b); err != nil {
				return err
			}
		}
		return nil
	}
}

func (m *measurer) batchedPtr(q *queue, batch int, blocks []unsafe.Pointer) func(int) error {
	return func(k int) error {
		issued := 0
		for done := 0; done < k; {
			b := min(batch, k-done)
			for i := 0; i < b; i++ {
				q.addPtr(blocks[i], m.n)
			}
			if err := q.flush(); err != nil {
				return err
			}
			done += b
			if err := m.maybeSync(&issued, b); err != nil {
				return err
			}
		}
		return nil
	}
}

// launchOnly issues saxpys with no host memory involved, batch at a time.
func (m *measurer) launchOnly(batch int) func(int) error {
	return func(k int) error {
		issued := 0
		for done := 0; done < k; {
			b := min(batch, k-done)
			var err error
			if b == 1 {
				err = check(C.lb_launch(C.int32_t(m.n)))
			} else {
				err = check(C.lb_launch_n(C.int32_t(m.n), C.int32_t(b)))
			}
			if err != nil {
				return err
			}
			done += b
			if err := m.maybeSync(&issued, b); err != nil {
				return err
			}
		}
		return nil
	}
}

// graph replays the current K-kernel graph from Go; k is a multiple of K.
func (m *measurer) graph(kernels int) func(int) error {
	return func(k int) error {
		issued := 0
		for done := 0; done < k; done += kernels {
			if err := check(C.lb_graph_launch()); err != nil {
				return err
			}
			if err := m.maybeSync(&issued, kernels); err != nil {
				return err
			}
		}
		return nil
	}
}

// queue is the launch-descriptor buffer: C memory, so the descriptors
// themselves are never moved by the collector, filled from Go and submitted
// in one crossing.
type queue struct {
	cmds []C.lcmd
	buf  unsafe.Pointer
	n    int
	pin  runtime.Pinner
}

func newQueue(capacity int) (*queue, error) {
	p := C.malloc(C.size_t(capacity) * C.size_t(unsafe.Sizeof(C.lcmd{})))
	if p == nil {
		return nil, errors.New("malloc failed")
	}
	return &queue{buf: p, cmds: unsafe.Slice((*C.lcmd)(p), capacity)}, nil
}

// addHeap stores a Go heap pointer into C memory, which the cgo rules allow
// only while the pointee is pinned. This Pin is the cost being measured.
func (q *queue) addHeap(block []float32, n int) {
	q.pin.Pin(&block[0])
	c := &q.cmds[q.n]
	c.host = (*C.float)(unsafe.Pointer(&block[0]))
	c.alpha = 1
	c.n = C.int32_t(n)
	q.n++
}

func (q *queue) addPtr(block unsafe.Pointer, n int) {
	c := &q.cmds[q.n]
	c.host = (*C.float)(block)
	c.alpha = 1
	c.n = C.int32_t(n)
	q.n++
}

func (q *queue) flush() error {
	if q.n == 0 {
		return nil
	}
	err := check(C.lb_batch(&q.cmds[0], C.int32_t(q.n)))
	q.n = 0
	// Pageable copies are staged before cudaMemcpyAsync returns, so the
	// host blocks may move again as soon as the batch call is back.
	q.pin.Unpin()
	return err
}

func (q *queue) close() {
	q.pin.Unpin()
	if q.buf != nil {
		C.free(q.buf)
		q.buf = nil
		q.cmds = nil
	}
}

// newBlocks returns count off-heap host blocks of n ones each, each aligned to
// align bytes. With align equal to the page size, every block gets a page of
// its own.
func newBlocks(count, n, align int) (*memory.Arena, []unsafe.Pointer, error) {
	a, err := memory.NewArena(count*max(n*4, align) + 2*align + 4096)
	if err != nil {
		return nil, nil, err
	}
	if !a.OffHeap() {
		a.Close()
		return nil, nil, errors.New("this platform has no off-heap arena; the comparison would be meaningless")
	}
	blocks := make([]unsafe.Pointer, count)
	for i := range blocks {
		b, err := a.Bytes(n*4, align)
		if err != nil {
			a.Close()
			return nil, nil, err
		}
		p := unsafe.Pointer(&b[0])
		ones(unsafe.Slice((*float32)(p), n))
		blocks[i] = p
	}
	return a, blocks, nil
}

// pageSpan returns the page-aligned range covering every block. The arena's
// mmap base is page-aligned and the OS maps whole pages, so rounding the ends
// outward never leaves the mapping.
func pageSpan(blocks []unsafe.Pointer, blockBytes int) (unsafe.Pointer, uintptr) {
	page := uintptr(os.Getpagesize())
	lo, hi := ^uintptr(0), uintptr(0)
	for _, b := range blocks {
		p := uintptr(b)
		if p < lo {
			lo = p
		}
		if e := p + uintptr(blockBytes); e > hi {
			hi = e
		}
	}
	lo &^= page - 1
	hi = (hi + page - 1) &^ (page - 1)
	// The arithmetic stays in uintptr; converting back is valid because the
	// address is inside a live mmap region the collector never manages.
	return unsafe.Add(blocks[0], -int(uintptr(blocks[0])-lo)), hi - lo
}

func ones(s []float32) []float32 {
	for i := range s {
		s[i] = 1
	}
	return s
}

func parseBatches(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		var b int
		if _, err := fmt.Sscanf(strings.TrimSpace(f), "%d", &b); err != nil || b <= 0 {
			return nil, fmt.Errorf("bad batch size %q", f)
		}
		out = append(out, b)
	}
	if len(out) == 0 {
		return nil, errors.New("no batch sizes")
	}
	return out, nil
}

func cudaVersion(v int) string {
	if v <= 0 {
		return "unknown"
	}
	return fmt.Sprintf("%d.%d", v/1000, (v%1000)/10)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func printTable(rep report) {
	h := rep.Host
	fmt.Printf("%s — CUDA driver %s, runtime %s — %s %s/%s, %d CPUs, GOMAXPROCS=%d, GODEBUG=%q, locked thread=%v\n\n",
		h.DeviceName, h.CUDADriver, h.CUDARuntime, h.GoVersion, h.GOOS, h.GOARCH, h.NumCPU,
		h.GOMAXPROCS, h.GODEBUG, h.LockedThread)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "measurement\tvalue\tunit\tnote")
	for _, f := range rep.Findings {
		fmt.Fprintf(w, "%s\t%.1f\t%s\t%s\n", f.Name, f.Value, f.Unit, f.Note)
	}
	w.Flush()
}
