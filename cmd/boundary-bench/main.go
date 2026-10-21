// Command boundary-bench runs every measurement in this repository and prints
// one table.
//
// The point is not to produce impressive numbers. It is to produce numbers you
// can check on your own machine, since the argument this repository makes -
// that Go should own the control plane and stay out of the data plane - is only
// worth anything if the measurements behind it reproduce.
//
// Usage:
//
//	go run ./cmd/boundary-bench                 # everything, ~30s
//	go run ./cmd/boundary-bench -quick          # smaller iteration counts
//	go run ./cmd/boundary-bench -json           # machine readable
//	go run ./cmd/boundary-bench -only=kernel,gc # a subset
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/pkg/device"
	"github.com/udaykishore-resu/gpu-boundary/pkg/ffi"
	"github.com/udaykishore-resu/gpu-boundary/pkg/gcprobe"
	"github.com/udaykishore-resu/gpu-boundary/pkg/kernel"
	"github.com/udaykishore-resu/gpu-boundary/pkg/kernel/cref"
	"github.com/udaykishore-resu/gpu-boundary/pkg/memory"
)

// Finding is one measured row. Keeping the output in a single flat shape makes
// it trivial to diff two runs, which is what you actually want when you change
// something.
type Finding struct {
	Section string  `json:"section"`
	Name    string  `json:"name"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
	Note    string  `json:"note"`
}

type report struct {
	Host     hostInfo  `json:"host"`
	Findings []Finding `json:"findings"`
}

type hostInfo struct {
	GOOS       string   `json:"goos"`
	GOARCH     string   `json:"goarch"`
	NumCPU     int      `json:"num_cpu"`
	GoVersion  string   `json:"go_version"`
	CGO        bool     `json:"cgo"`
	Kernels    []string `json:"kernels"`
	Devices    []string `json:"devices"`
	DeviceName string   `json:"device_name"`
}

var (
	flagJSON  = flag.Bool("json", false, "emit JSON instead of a table")
	flagQuick = flag.Bool("quick", false, "run with reduced iteration counts")
	flagOnly  = flag.String("only", "", "comma separated sections: boundary,launch,stall,kernel,gc,device")
)

func main() {
	flag.Parse()

	sections := map[string]bool{}
	if *flagOnly != "" {
		for _, s := range strings.Split(*flagOnly, ",") {
			sections[strings.TrimSpace(s)] = true
		}
	}
	want := func(s string) bool { return len(sections) == 0 || sections[s] }

	rep := report{Host: collectHost()}

	if want("boundary") {
		rep.Findings = append(rep.Findings, boundarySection()...)
	}
	if want("launch") {
		rep.Findings = append(rep.Findings, launchSection()...)
	}
	if want("stall") {
		rep.Findings = append(rep.Findings, stallSection()...)
	}
	if want("kernel") {
		rep.Findings = append(rep.Findings, kernelSection()...)
	}
	if want("gc") {
		rep.Findings = append(rep.Findings, gcSection()...)
	}
	if want("device") {
		rep.Findings = append(rep.Findings, deviceSection()...)
	}

	if *flagJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	printTable(rep)
}

func collectHost() hostInfo {
	h := hostInfo{
		GOOS:      runtime.GOOS,
		GOARCH:    runtime.GOARCH,
		NumCPU:    runtime.NumCPU(),
		GoVersion: runtime.Version(),
		CGO:       ffi.Available,
	}
	for _, i := range kernel.Impls() {
		h.Kernels = append(h.Kernels, i.Name)
	}
	if cref.Available {
		h.Kernels = append(h.Kernels, "cgo-c")
	}
	for _, k := range device.Registered() {
		h.Devices = append(h.Devices, string(k))
	}
	if d, err := device.OpenBest(); err == nil {
		h.DeviceName = d.Name()
		d.Close()
	}
	return h
}

func iters(full, quick int) int {
	if *flagQuick {
		return quick
	}
	return full
}

// ---------------------------------------------------------------- boundary

func boundarySection() []Finding {
	const sec = "cgo boundary"
	if !ffi.Available {
		return []Finding{{Section: sec, Name: "cgo", Note: "built with CGO_ENABLED=0, nothing to measure"}}
	}

	n := iters(3_000_000, 300_000)

	goNS := timeLoop(n, func() { goNoop() })
	cgoNS := timeLoop(n, func() { ffi.Noop() })
	addNS := timeLoop(n, func() { ffi.Add(1, 2) })

	return []Finding{
		{sec, "go function call", goNS, "ns/op", "//go:noinline empty Go function"},
		{sec, "cgo call", cgoNS, "ns/op", "empty C function"},
		{sec, "cgo call with args", addNS, "ns/op", "one int32 addition in C"},
		{sec, "boundary tax", cgoNS - goNS, "ns/call", "what every GPU API call pays before doing anything"},
		{sec, "tax ratio", safeDiv(cgoNS, goNS), "x", "cgo call cost relative to a Go call"},
	}
}

//go:noinline
func goNoop() {}

// ------------------------------------------------------------------ launch

func launchSection() []Finding {
	const sec = "launch batching"
	if !ffi.Available {
		return nil
	}

	out := []Finding{}
	total := iters(200_000, 20_000)
	const elems = 16 // a tiny kernel, so the boundary dominates the arithmetic

	// Heap path: Go slices, so every descriptor has to pin its buffers.
	hx := make([]float32, elems)
	hy := make([]float32, elems)

	// Off-heap path: arena memory the collector never sees, so no pinning.
	arena, err := memory.NewArena(1 << 16)
	if err != nil {
		return append(out, Finding{sec, "arena", 0, "", err.Error()})
	}
	defer arena.Close()
	ax := arena.MustFloat32(elems)
	ay := arena.MustFloat32(elems)
	axp, ayp := memory.Pointer(ax), memory.Pointer(ay)

	var heapBase, arenaBase float64
	for _, batch := range []int{1, 4, 16, 64, 256} {
		q, err := ffi.NewQueue(batch)
		if err != nil {
			return append(out, Finding{sec, "error", 0, "", err.Error()})
		}

		heapNS := timeLoop(total, func() {
			q.Enqueue(ffi.OpSaxpy, 1.0000001, hx, hy)
			if q.Len() == batch {
				q.Flush()
			}
		})
		q.Flush()

		arenaNS := timeLoop(total, func() {
			q.EnqueuePtr(ffi.OpSaxpy, 1.0000001, axp, ayp, elems)
			if q.Len() == batch {
				q.Flush()
			}
		})
		q.Flush()
		q.Close()

		if batch == 1 {
			heapBase, arenaBase = heapNS, arenaNS
		}

		heapNote := fmt.Sprintf("%d per crossing, pinned Go memory", batch)
		arenaNote := fmt.Sprintf("%d per crossing, off-heap memory", batch)
		if batch > 1 {
			heapNote += fmt.Sprintf(", %.2fx vs unbatched", safeDiv(heapBase, heapNS))
			arenaNote += fmt.Sprintf(", %.2fx vs unbatched", safeDiv(arenaBase, arenaNS))
		}
		out = append(out,
			Finding{sec, fmt.Sprintf("batch=%-3d heap", batch), heapNS, "ns/launch", heapNote},
			Finding{sec, fmt.Sprintf("batch=%-3d arena", batch), arenaNS, "ns/launch", arenaNote},
		)
	}
	out = append(out, Finding{sec, "pinning cost", heapBase - arenaBase, "ns/descriptor",
		"runtime.Pinner per buffer; this is why batching alone does not pay"})
	return out
}

// ------------------------------------------------------------------- stall

func stallSection() []Finding {
	const sec = "scheduler"
	if !ffi.Available {
		return nil
	}

	window := 400 * time.Millisecond
	if *flagQuick {
		window = 150 * time.Millisecond
	}
	blockers := runtime.GOMAXPROCS(0) * 4

	inGo := ffi.MeasureStall(ffi.StallConfig{
		Blockers: blockers, Hold: 2 * time.Millisecond, Window: window, Tick: time.Millisecond, InC: false,
	})
	inC := ffi.MeasureStall(ffi.StallConfig{
		Blockers: blockers, Hold: 2 * time.Millisecond, Window: window, Tick: time.Millisecond, InC: true,
	})

	return []Finding{
		{sec, "threads created, blocking in Go", float64(inGo.ThreadGrowth()), "threads",
			fmt.Sprintf("%d goroutines parked by the runtime", blockers)},
		{sec, "threads created, blocking in C", float64(inC.ThreadGrowth()), "threads",
			"the runtime cannot park a goroutine inside a C call, so it makes a thread"},
		{sec, "observer p99 lateness, blocking in Go", float64(inGo.LateP99.Microseconds()), "us", ""},
		{sec, "observer p99 lateness, blocking in C", float64(inC.LateP99.Microseconds()), "us",
			"a synchronous GPU API called from many goroutines behaves like this"},
	}
}

// ------------------------------------------------------------------ kernel

func kernelSection() []Finding {
	const sec = "cpu kernel"
	out := []Finding{}

	for _, n := range []int{256, 4096, 65536} {
		a := ramp(n, 0.5)
		b := ramp(n, -0.25)

		type run struct {
			name string
			fn   func([]float32, []float32) float32
		}
		runs := []run{}
		for _, impl := range kernel.Impls() {
			runs = append(runs, run{impl.Name, impl.Dot})
		}
		if cref.Available {
			runs = append(runs, run{"cgo-c", cref.Dot})
		}

		var scalarNS float64
		for _, r := range runs {
			ns := timeBudget(kernelBudget(), func() { sink = r.fn(a, b) })
			if r.name == "scalar" {
				scalarNS = ns
			}
			gbs := 2 * float64(n) * 4 / ns // bytes per ns == GB/s
			note := fmt.Sprintf("%.1f GB/s", gbs)
			if scalarNS > 0 && r.name != "scalar" {
				note += fmt.Sprintf(", %.2fx over scalar", safeDiv(scalarNS, ns))
			}
			out = append(out, Finding{sec, fmt.Sprintf("n=%d %s", n, r.name), ns, "ns/op", note})
		}
	}
	return out
}

var sink float32

// kernelBudget is how long each kernel measurement runs. v0.2.0 used a fixed
// iteration count, which at n=256 finished in under a millisecond and varied by
// 30% or more between runs; a time budget keeps every row long enough to be
// stable regardless of how fast the kernel or the machine is.
func kernelBudget() time.Duration {
	if *flagQuick {
		return 20 * time.Millisecond
	}
	return 200 * time.Millisecond
}

// timeBudget calibrates an iteration count that fills about d, then measures
// with timeLoop.
func timeBudget(d time.Duration, fn func()) float64 {
	for n := 1; ; n *= 2 {
		start := time.Now()
		for i := 0; i < n; i++ {
			fn()
		}
		if el := time.Since(start); el >= d/10 || n >= 1<<30 {
			reps := int(float64(n) * float64(d) / float64(max(el, time.Nanosecond)))
			return timeLoop(max(reps, 1), fn)
		}
	}
}

func ramp(n int, step float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(i%251)*step - 3
	}
	return s
}

// ---------------------------------------------------------------------- gc

func gcSection() []Finding {
	const sec = "allocation"
	out := []Finding{}

	cfg := gcprobe.Config{
		Elems:      8192,
		Iterations: iters(30000, 4000),
	}
	results, err := gcprobe.RunAll(cfg)
	if err != nil {
		return append(out, Finding{sec, "error", 0, "", err.Error()})
	}

	var heapP99 float64
	for _, r := range results {
		name := string(r.Config.Strategy)
		if name == "heap" {
			heapP99 = float64(r.P99.Nanoseconds())
		}
		note := fmt.Sprintf("%.1f MB allocated, %d collections", r.TotalAllocMB, r.NumGC)
		if heapP99 > 0 && name != "heap" {
			note += fmt.Sprintf(", p99 %.2fx better than heap", safeDiv(heapP99, float64(r.P99.Nanoseconds())))
		}
		out = append(out,
			Finding{sec, name + " p50", float64(r.P50.Nanoseconds()), "ns", ""},
			Finding{sec, name + " p99", float64(r.P99.Nanoseconds()), "ns", note},
			Finding{sec, name + " p99.9", float64(r.P999.Nanoseconds()), "ns", ""},
			Finding{sec, name + " max", float64(r.Max.Nanoseconds()), "ns", ""},
		)
	}
	return out
}

// ------------------------------------------------------------------ device

func deviceSection() []Finding {
	const sec = "device"
	d, err := device.OpenBest()
	if err != nil {
		return []Finding{{sec, "open", 0, "", err.Error()}}
	}
	defer d.Close()

	const n = 4096
	x, err := d.Alloc(n)
	if err != nil {
		return []Finding{{sec, "alloc", 0, "", err.Error()}}
	}
	defer x.Free()
	y, err := d.Alloc(n)
	if err != nil {
		return []Finding{{sec, "alloc", 0, "", err.Error()}}
	}
	defer y.Free()

	host := ramp(n, 0.1)
	if err := d.Upload(x, host); err != nil {
		return []Finding{{sec, "upload", 0, "", err.Error()}}
	}
	if err := d.Upload(y, host); err != nil {
		return []Finding{{sec, "upload", 0, "", err.Error()}}
	}

	note := d.Name()
	if d.Kind() == device.KindCPU {
		note += " - no driver to submit to, so batching changes the counters but not the clock; " +
			"build with -tags metal or -tags cuda for the real comparison"
	}
	out := []Finding{{sec, "backend", 0, "", note}}
	total := iters(20000, 2000)

	var base float64
	for _, batch := range []int{1, 8, 64} {
		start := time.Now()
		for i := 0; i < total/batch; i++ {
			for k := 0; k < batch; k++ {
				if err := d.Saxpy(1.0000001, x, y); err != nil {
					return append(out, Finding{sec, "saxpy", 0, "", err.Error()})
				}
			}
			if err := d.Sync(); err != nil {
				return append(out, Finding{sec, "sync", 0, "", err.Error()})
			}
		}
		per := float64(time.Since(start).Nanoseconds()) / float64((total/batch)*batch)
		if batch == 1 {
			base = per
		}
		note := fmt.Sprintf("%d launches per submission", batch)
		if batch > 1 {
			note += fmt.Sprintf(", %.2fx vs unbatched", safeDiv(base, per))
		}
		out = append(out, Finding{sec, fmt.Sprintf("saxpy batch=%d", batch), per, "ns/launch", note})
	}

	st := d.Stats()
	out = append(out, Finding{sec, "amplification", st.Amplification(), "launches/submission",
		fmt.Sprintf("%d enqueued over %d submissions", st.Enqueued, st.Submissions)})
	return out
}

// ------------------------------------------------------------------ output

func timeLoop(n int, fn func()) float64 {
	if n <= 0 {
		return 0
	}
	// A short warmup so the first measurement is not paying for page faults
	// and branch predictor cold start.
	for i := 0; i < n/20+1; i++ {
		fn()
	}
	start := time.Now()
	for i := 0; i < n; i++ {
		fn()
	}
	return float64(time.Since(start).Nanoseconds()) / float64(n)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func printTable(rep report) {
	h := rep.Host
	fmt.Printf("gpu-boundary measurements\n")
	fmt.Printf("%s/%s, %d cpus, %s, cgo=%v\n", h.GOOS, h.GOARCH, h.NumCPU, h.GoVersion, h.CGO)
	fmt.Printf("kernels: %s\n", strings.Join(h.Kernels, ", "))
	fmt.Printf("devices: %s", strings.Join(h.Devices, ", "))
	if h.DeviceName != "" {
		fmt.Printf(" (best: %s)", h.DeviceName)
	}
	fmt.Printf("\n\n")

	sections := []string{}
	bySection := map[string][]Finding{}
	for _, f := range rep.Findings {
		if _, ok := bySection[f.Section]; !ok {
			sections = append(sections, f.Section)
		}
		bySection[f.Section] = append(bySection[f.Section], f)
	}
	sort.SliceStable(sections, func(i, j int) bool { return false }) // preserve order

	for _, s := range sections {
		fmt.Printf("%s\n%s\n", s, strings.Repeat("-", len(s)))
		w := 0
		for _, f := range bySection[s] {
			if len(f.Name) > w {
				w = len(f.Name)
			}
		}
		for _, f := range bySection[s] {
			val := ""
			if f.Unit != "" {
				val = fmt.Sprintf("%12s %-20s", formatValue(f.Value), f.Unit)
			} else {
				val = fmt.Sprintf("%12s %-20s", "", "")
			}
			fmt.Printf("  %-*s  %s %s\n", w, f.Name, val, f.Note)
		}
		fmt.Println()
	}
}

func formatValue(v float64) string {
	switch {
	case v == 0:
		return "0"
	case v >= 1000:
		return fmt.Sprintf("%.0f", v)
	case v >= 10:
		return fmt.Sprintf("%.1f", v)
	default:
		return fmt.Sprintf("%.2f", v)
	}
}
