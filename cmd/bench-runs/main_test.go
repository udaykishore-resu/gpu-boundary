package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// When this variable is set the test binary pretends to be a measurement
// command, so the end-to-end test exercises real process spawning without
// depending on any other binary being built.
const fakeEnv = "BENCH_RUNS_FAKE_CHILD"

func TestMain(m *testing.M) {
	if os.Getenv(fakeEnv) == "1" {
		fmt.Print(`{"host":{"goos":"linux","goarch":"amd64","num_cpu":4,"go_version":"go1.24"},` +
			`"findings":[{"section":"boundary","name":"cgo call","value":66.5,"unit":"ns/call","note":""},` +
			`{"section":"gc","name":"p99","value":30.0,"unit":"µs","note":"heap"}]}`)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestStats(t *testing.T) {
	cases := []struct {
		in                  []float64
		median, min, max    float64
		mean, stddev, cvPct float64
	}{
		{in: []float64{5}, median: 5, min: 5, max: 5, mean: 5},
		{in: []float64{4, 1, 3, 2}, median: 2.5, min: 1, max: 4, mean: 2.5,
			stddev: math.Sqrt(5.0 / 3.0), cvPct: 100 * math.Sqrt(5.0/3.0) / 2.5},
		{in: []float64{10, 10, 10}, median: 10, min: 10, max: 10, mean: 10},
	}
	for _, c := range cases {
		s := stats(c.in)
		for name, pair := range map[string][2]float64{
			"median": {s.Median, c.median}, "min": {s.Min, c.min}, "max": {s.Max, c.max},
			"mean": {s.Mean, c.mean}, "stddev": {s.StdDev, c.stddev}, "cv": {s.CVPct, c.cvPct},
		} {
			if math.Abs(pair[0]-pair[1]) > 1e-9 {
				t.Errorf("stats(%v) %s = %v, want %v", c.in, name, pair[0], pair[1])
			}
		}
		if s.N != len(c.in) {
			t.Errorf("stats(%v) n = %d", c.in, s.N)
		}
	}
}

func TestStatsDoesNotReorderInput(t *testing.T) {
	in := []float64{3, 1, 2}
	stats(in)
	if in[0] != 3 || in[1] != 1 || in[2] != 2 {
		t.Fatalf("input was mutated: %v", in)
	}
}

func TestSummariseKeepsFirstSeenOrderAndCountsMissing(t *testing.T) {
	raw := [][]Finding{
		{{Section: "a", Name: "x", Value: 1}, {Section: "b", Name: "y", Value: 10}},
		{{Section: "a", Name: "x", Value: 3}},
	}
	got := summarise(raw)
	if len(got) != 2 || got[0].Section != "a" || got[1].Section != "b" {
		t.Fatalf("order: %+v", got)
	}
	if got[0].N != 2 || got[0].Median != 2 {
		t.Fatalf("a/x: %+v", got[0])
	}
	if got[1].N != 1 {
		t.Fatalf("b/y should be seen once: %+v", got[1])
	}
}

func TestSanitise(t *testing.T) {
	for in, want := range map[string]string{
		"m3 air":        "m3-air",
		"../../etc":     "etc",
		"":              "host",
		"aws g6.xlarge": "aws-g6.xlarge",
	} {
		if got := sanitise(in); got != want {
			t.Errorf("sanitise(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEndToEnd(t *testing.T) {
	t.Setenv(fakeEnv, "1")
	dir := t.TempDir()
	var stdout, stderr bytes.Buffer
	err := run([]string{"-runs", "3", "-cooldown", "0", "-out", dir, "-label", "test box",
		"--", os.Args[0]}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run: %v\nstderr:\n%s", err, stderr.String())
	}

	files, _ := filepath.Glob(filepath.Join(dir, "test-box-*.json"))
	if len(files) != 1 {
		t.Fatalf("want one results file, got %v", files)
	}
	b, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	var res Results
	if err := json.Unmarshal(b, &res); err != nil {
		t.Fatal(err)
	}
	if res.Schema != Schema || res.Runs != 3 || len(res.Raw) != 3 || len(res.Summary) != 2 {
		t.Fatalf("unexpected results: schema=%q runs=%d raw=%d summary=%d",
			res.Schema, res.Runs, len(res.Raw), len(res.Summary))
	}
	if s := res.Summary[0]; s.N != 3 || s.Median != 66.5 || s.StdDev != 0 {
		t.Fatalf("summary[0] = %+v", s)
	}
	if len(res.Warnings) != 0 {
		t.Fatalf("unexpected warnings: %v", res.Warnings)
	}
	if !strings.Contains(stdout.String(), "| boundary | cgo call | 66.5 |") {
		t.Fatalf("table missing row:\n%s", stdout.String())
	}
}

func TestRejectsNonJSONOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{"-runs", "1", "-cooldown", "0", "-out", t.TempDir(), "--", "echo", "not json"},
		&stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "JSON") {
		t.Fatalf("want a JSON parse error, got %v", err)
	}
}
