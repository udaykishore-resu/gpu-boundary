// Command bench-runs turns a single benchmark run into evidence.
//
// One run of boundary-bench is an anecdote: it says what one process saw on
// one machine in one thermal state. A number that goes into a paper needs a
// distribution behind it, a record of the hardware it came from, and the raw
// data kept next to the summary so that anyone can recompute it.
//
// bench-runs executes a measurement command N times, each in a fresh process
// so that one run's heap, page cache and JIT-free-but-still-warm branch
// predictors cannot leak into the next, parses the JSON every run emits, and
// writes one results file containing:
//
//   - the host and machine description (CPU model, kernel, GPU and driver when
//     present),
//   - the exact command line,
//   - per-finding summary statistics across runs (median, min, max, mean,
//     sample standard deviation, coefficient of variation),
//   - every run's raw findings, unmodified.
//
// Usage:
//
//	go run ./cmd/bench-runs -runs 5 -- bin/boundary-bench -json
//	go run ./cmd/bench-runs -runs 5 -label m3-air -- bin/boundary-bench -json
//	go run ./cmd/bench-runs -runs 5 -- bin/cuda-launch-bench -json
//
// Any command works as long as it prints the boundary-bench report shape
// ({"host": {...}, "findings": [{section, name, value, unit, note}, ...]}) on
// stdout. The summary table is printed as Markdown so it can be pasted into a
// draft without reformatting.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// Schema identifies the results file format, so a later analysis script can
// refuse a file it does not understand instead of misreading it.
const Schema = "gpu-boundary/bench-runs/v1"

// Finding is one measured row, in the shape every measurement command in this
// repository emits.
type Finding struct {
	Section string  `json:"section"`
	Name    string  `json:"name"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
	Note    string  `json:"note"`
}

type runReport struct {
	Host     json.RawMessage `json:"host"`
	Findings []Finding       `json:"findings"`
}

// Summary is the cross-run statistics for one finding.
type Summary struct {
	Section string  `json:"section"`
	Name    string  `json:"name"`
	Unit    string  `json:"unit"`
	Note    string  `json:"note"`
	N       int     `json:"n"`
	Median  float64 `json:"median"`
	Min     float64 `json:"min"`
	Max     float64 `json:"max"`
	Mean    float64 `json:"mean"`
	StdDev  float64 `json:"stddev"`
	CVPct   float64 `json:"cv_pct"`
}

// Machine records what the measurement command cannot see from inside the Go
// runtime: the CPU model, the kernel and, when present, the GPU and driver.
type Machine struct {
	CPUModel string `json:"cpu_model,omitempty"`
	Kernel   string `json:"kernel,omitempty"`
	GPU      string `json:"gpu,omitempty"`
}

// Results is the file bench-runs writes.
type Results struct {
	Schema     string          `json:"schema"`
	Label      string          `json:"label"`
	CreatedUTC string          `json:"created_utc"`
	Command    []string        `json:"command"`
	Runs       int             `json:"runs"`
	Host       json.RawMessage `json:"host"`
	Machine    Machine         `json:"machine"`
	Warnings   []string        `json:"warnings,omitempty"`
	Summary    []Summary       `json:"summary"`
	Raw        [][]Finding     `json:"raw"`
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "bench-runs:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("bench-runs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runs := fs.Int("runs", 5, "number of runs, each in a fresh process")
	out := fs.String("out", "results", "directory to write the results file into")
	label := fs.String("label", "", "name for this machine (default goos-goarch-Ncpu from the report)")
	cooldown := fs.Duration("cooldown", 2*time.Second, "pause between runs, to let clocks and thermals settle")
	quiet := fs.Bool("quiet", false, "do not print the summary table")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: bench-runs [flags] -- command [args...]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	cmdline := fs.Args()
	if len(cmdline) == 0 {
		fs.Usage()
		return errors.New("no command given")
	}
	if *runs < 1 {
		return errors.New("-runs must be at least 1")
	}

	res := Results{
		Schema:     Schema,
		CreatedUTC: time.Now().UTC().Format(time.RFC3339),
		Command:    cmdline,
		Runs:       *runs,
		Machine:    describeMachine(),
	}

	for i := 0; i < *runs; i++ {
		if i > 0 && *cooldown > 0 {
			time.Sleep(*cooldown)
		}
		fmt.Fprintf(stderr, "bench-runs: run %d/%d: %s\n", i+1, *runs, strings.Join(cmdline, " "))
		rep, err := runOnce(cmdline, stderr)
		if err != nil {
			return fmt.Errorf("run %d: %w", i+1, err)
		}
		if res.Host == nil {
			res.Host = rep.Host
		} else if !bytes.Equal(compactJSON(res.Host), compactJSON(rep.Host)) {
			res.Warnings = append(res.Warnings,
				fmt.Sprintf("run %d reported a different host description than run 1", i+1))
		}
		res.Raw = append(res.Raw, rep.Findings)
	}

	res.Summary = summarise(res.Raw)
	for _, s := range res.Summary {
		if s.N != *runs {
			res.Warnings = append(res.Warnings,
				fmt.Sprintf("%s / %s appeared in %d of %d runs", s.Section, s.Name, s.N, *runs))
		}
	}

	res.Label = *label
	if res.Label == "" {
		res.Label = defaultLabel(res.Host)
	}
	res.Label = sanitise(res.Label)

	path, err := write(*out, res)
	if err != nil {
		return err
	}
	if !*quiet {
		printTable(stdout, res)
	}
	for _, w := range res.Warnings {
		fmt.Fprintln(stderr, "bench-runs: warning:", w)
	}
	fmt.Fprintln(stderr, "bench-runs: wrote", path)
	return nil
}

func runOnce(cmdline []string, stderr io.Writer) (runReport, error) {
	var stdout bytes.Buffer
	c := exec.Command(cmdline[0], cmdline[1:]...)
	c.Stdout = &stdout
	c.Stderr = stderr
	if err := c.Run(); err != nil {
		return runReport{}, err
	}
	var rep runReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		return runReport{}, fmt.Errorf("parsing output as JSON (did you pass -json?): %w", err)
	}
	if len(rep.Findings) == 0 {
		return runReport{}, errors.New("command reported no findings")
	}
	return rep, nil
}

// summarise groups findings by (section, name) across runs, preserving the
// order in which they first appear, and computes the cross-run statistics.
func summarise(raw [][]Finding) []Summary {
	type acc struct {
		first Finding
		vals  []float64
	}
	byKey := map[string]*acc{}
	var order []string
	for _, run := range raw {
		for _, f := range run {
			k := f.Section + "\x00" + f.Name
			a, ok := byKey[k]
			if !ok {
				a = &acc{first: f}
				byKey[k] = a
				order = append(order, k)
			}
			a.vals = append(a.vals, f.Value)
		}
	}
	out := make([]Summary, 0, len(order))
	for _, k := range order {
		a := byKey[k]
		s := stats(a.vals)
		s.Section, s.Name, s.Unit, s.Note = a.first.Section, a.first.Name, a.first.Unit, a.first.Note
		out = append(out, s)
	}
	return out
}

// stats computes order statistics and moments for one finding. The standard
// deviation is the sample (n-1) form, because the runs are a sample of what
// the machine does, not the whole population.
func stats(vals []float64) Summary {
	s := Summary{N: len(vals)}
	if len(vals) == 0 {
		return s
	}
	v := append([]float64(nil), vals...)
	sort.Float64s(v)
	s.Min, s.Max = v[0], v[len(v)-1]
	if len(v)%2 == 1 {
		s.Median = v[len(v)/2]
	} else {
		s.Median = (v[len(v)/2-1] + v[len(v)/2]) / 2
	}
	sum := 0.0
	for _, x := range v {
		sum += x
	}
	s.Mean = sum / float64(len(v))
	if len(v) > 1 {
		ss := 0.0
		for _, x := range v {
			d := x - s.Mean
			ss += d * d
		}
		s.StdDev = math.Sqrt(ss / float64(len(v)-1))
	}
	if s.Mean != 0 {
		s.CVPct = 100 * s.StdDev / math.Abs(s.Mean)
	}
	return s
}

func write(dir string, res Results) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	stamp := strings.NewReplacer("-", "", ":", "").Replace(res.CreatedUTC)
	path := filepath.Join(dir, fmt.Sprintf("%s-%s.json", res.Label, stamp))
	b, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return "", err
	}
	b = append(b, '\n')
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func printTable(w io.Writer, res Results) {
	fmt.Fprintf(w, "\n%s — %d runs — %s\n", res.Label, res.Runs, res.CreatedUTC)
	if res.Machine.CPUModel != "" {
		fmt.Fprintf(w, "cpu: %s\n", res.Machine.CPUModel)
	}
	if res.Machine.GPU != "" {
		fmt.Fprintf(w, "gpu: %s\n", res.Machine.GPU)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| section | measurement | median | min | max | cv % | unit |")
	fmt.Fprintln(w, "|---|---|---:|---:|---:|---:|---|")
	for _, s := range res.Summary {
		fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %.1f | %s |\n",
			s.Section, strings.TrimSpace(s.Name), num(s.Median), num(s.Min), num(s.Max), s.CVPct, s.Unit)
	}
	fmt.Fprintln(w)
}

func num(v float64) string {
	switch a := math.Abs(v); {
	case a == 0:
		return "0"
	case a >= 1000:
		return fmt.Sprintf("%.0f", v)
	case a >= 10:
		return fmt.Sprintf("%.1f", v)
	default:
		return fmt.Sprintf("%.3g", v)
	}
}

func compactJSON(b []byte) []byte {
	var buf bytes.Buffer
	if err := json.Compact(&buf, b); err != nil {
		return b
	}
	return buf.Bytes()
}

func defaultLabel(host json.RawMessage) string {
	var h struct {
		GOOS   string `json:"goos"`
		GOARCH string `json:"goarch"`
		NumCPU int    `json:"num_cpu"`
	}
	if err := json.Unmarshal(host, &h); err != nil || h.GOOS == "" {
		return fmt.Sprintf("%s-%s-%dcpu", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	}
	return fmt.Sprintf("%s-%s-%dcpu", h.GOOS, h.GOARCH, h.NumCPU)
}

var unsafeLabel = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

func sanitise(s string) string {
	s = unsafeLabel.ReplaceAllString(strings.TrimSpace(s), "-")
	s = strings.Trim(s, "-.")
	if s == "" {
		return "host"
	}
	return s
}

// describeMachine collects what it can without failing: every probe is best
// effort, because a missing nvidia-smi on a laptop is not an error.
func describeMachine() Machine {
	m := Machine{
		CPUModel: cpuModel(),
		Kernel:   firstLine(output("uname", "-sr")),
	}
	if g := firstLine(output("nvidia-smi",
		"--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")); g != "" {
		m.GPU = g
	}
	return m
}

func cpuModel() string {
	switch runtime.GOOS {
	case "darwin":
		return firstLine(output("sysctl", "-n", "machdep.cpu.brand_string"))
	case "linux":
		b, err := os.ReadFile("/proc/cpuinfo")
		if err != nil {
			return ""
		}
		for _, line := range strings.Split(string(b), "\n") {
			k, v, ok := strings.Cut(line, ":")
			if !ok {
				continue
			}
			switch strings.TrimSpace(k) {
			case "model name", "Model", "Hardware":
				if v = strings.TrimSpace(v); v != "" {
					return v
				}
			}
		}
	}
	return ""
}

func output(name string, args ...string) string {
	if _, err := exec.LookPath(name); err != nil {
		return ""
	}
	b, err := exec.Command(name, args...).Output()
	if err != nil {
		return ""
	}
	return string(b)
}

func firstLine(s string) string {
	s, _, _ = strings.Cut(strings.TrimSpace(s), "\n")
	return strings.TrimSpace(s)
}
