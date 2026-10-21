// Command router-bench times the control-plane routing decision with exactly
// the harness bench/python/bench_router.py uses, so the two numbers are
// comparable, and prints the workload trace that proves both sides did the
// same work.
//
//	go run ./cmd/router-bench                # table
//	go run ./cmd/router-bench -json          # for cmd/bench-runs
//	go run ./cmd/router-bench -trace         # canonical trace JSON for the cross-check
//
// The timed loop is one Pick plus one Done per request, over the deterministic
// workload in internal/routerload: build the fleet, run one untimed warmup
// pass, then -repeats timed passes of -iters requests, and report the median
// pass. Go's garbage collector runs normally, as CPython's does on the other
// side; neither is tuned.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"text/tabwriter"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/internal/routerload"
)

type Finding struct {
	Section string  `json:"section"`
	Name    string  `json:"name"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
	Note    string  `json:"note"`
}

type hostInfo struct {
	GOOS      string `json:"goos"`
	GOARCH    string `json:"goarch"`
	NumCPU    int    `json:"num_cpu"`
	GoVersion string `json:"go_version"`
	Language  string `json:"language"`
}

type report struct {
	Host     hostInfo  `json:"host"`
	Findings []Finding `json:"findings"`
}

const keySpace = 1 << 16

func main() {
	jsonOut := flag.Bool("json", false, "emit JSON instead of a table")
	trace := flag.Bool("trace", false, "print the canonical workload trace and exit")
	traceN := flag.Int("trace-n", 100_000, "requests in the trace")
	iters := flag.Int("iters", 200_000, "requests per timed pass")
	repeats := flag.Int("repeats", 3, "timed passes; the median is reported")
	flag.Parse()

	if *trace {
		b, err := json.Marshal(routerload.Trace(routerload.Seed, *traceN, routerload.Replicas, routerload.Hold))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println(string(b))
		return
	}

	keys := routerload.Keys(routerload.Seed, keySpace)
	rep := report{Host: hostInfo{
		GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, NumCPU: runtime.NumCPU(),
		GoVersion: runtime.Version(), Language: "go",
	}}
	for _, replicas := range []int{8, 64} {
		ns := timePasses(keys, replicas, *iters, *repeats)
		rep.Findings = append(rep.Findings, Finding{
			Section: "router",
			Name:    fmt.Sprintf("go pick+done, replicas=%d", replicas),
			Value:   ns,
			Unit:    "ns/request",
			Note:    fmt.Sprintf("median of %d passes x %d requests", *repeats, *iters),
		})
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "measurement\tvalue\tunit\tnote")
	for _, f := range rep.Findings {
		fmt.Fprintf(w, "%s\t%.1f\t%s\t%s\n", f.Name, f.Value, f.Unit, f.Note)
	}
	w.Flush()
}

// timePasses mirrors time_passes in bench/python/bench_router.py.
func timePasses(keys []string, replicas, iters, repeats int) float64 {
	pass := func() time.Duration {
		win := routerload.NewWindow(routerload.NewRouter(replicas), routerload.Hold)
		start := time.Now()
		for i := 0; i < iters; i++ {
			win.Step(keys[i&(keySpace-1)])
		}
		d := time.Since(start)
		win.Drain()
		return d
	}
	pass() // warmup
	var ds []float64
	for r := 0; r < repeats; r++ {
		ds = append(ds, float64(pass().Nanoseconds())/float64(iters))
	}
	sort.Float64s(ds)
	return ds[len(ds)/2]
}
