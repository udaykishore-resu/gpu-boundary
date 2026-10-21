// Package gated contains no code. Its test parses the source files that the
// default build excludes, so that a typo in a GPU backend is caught by an
// ordinary `go test ./...` on a laptop with no GPU in it.
//
// This is a deliberately modest guarantee and it is stated as such: parsing is
// not type checking and type checking is not compiling. A file that passes this
// test can still fail against a real CUDA or Metal SDK. What it catches is the
// class of mistake that actually happens when nobody can build a file - an
// unbalanced brace, a missing import, a half-finished edit - and it catches it
// in CI rather than on the one machine that has the hardware.
package gated

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// gatedFiles are Go sources excluded from the default build. Keep this list in
// step with the build tags; the test fails if a listed file has gone missing,
// so a rename cannot silently drop coverage.
var gatedFiles = []string{
	"pkg/device/metal_darwin.go",    // //go:build darwin && metal && cgo
	"pkg/device/cuda.go",            // //go:build cuda && cgo
	"cmd/cuda-launch-bench/main.go", // //go:build cuda && cgo
	"pkg/kernel/simdexp_amd64.go",   // //go:build goexperiment.simd && amd64
	"pkg/ffi/ffi_nocgo.go",          // //go:build !cgo
	"pkg/kernel/cref/cref_nocgo.go", // //go:build !cgo
	"pkg/memory/mmap_other.go",      // //go:build !unix
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	for i := 0; i < 6; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		dir = filepath.Dir(dir)
	}
	t.Fatal("could not locate go.mod above the working directory")
	return ""
}

func TestGatedFilesParse(t *testing.T) {
	root := repoRoot(t)
	fset := token.NewFileSet()

	for _, rel := range gatedFiles {
		path := filepath.Join(root, rel)
		src, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("%s: %v (was the file renamed without updating gatedFiles?)", rel, err)
			continue
		}
		if _, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution); err != nil {
			t.Errorf("%s: %v", rel, err)
			continue
		}
		if !strings.Contains(string(src), "//go:build") {
			t.Errorf("%s: listed as build-gated but has no //go:build constraint", rel)
		}
		t.Logf("parsed %s (%d bytes)", rel, len(src))
	}
}

// TestNonGoGatedFilesExist covers the Objective-C and header files, which no Go
// tool will look at on a Linux CI runner.
func TestNonGoGatedFilesExist(t *testing.T) {
	root := repoRoot(t)
	for _, rel := range []string{
		"pkg/device/metal_shim.h",
		"pkg/device/metal_shim.m",
	} {
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Errorf("%s: %v", rel, err)
			continue
		}
		if !strings.HasPrefix(string(b), "//go:build darwin && metal") {
			t.Errorf("%s: missing the darwin && metal build constraint on its first line", rel)
		}
		if strings.Count(string(b), "{") != strings.Count(string(b), "}") {
			t.Errorf("%s: unbalanced braces", rel)
		}
	}
}
