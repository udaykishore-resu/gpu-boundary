//go:build !(cuda && cgo)

// Command cuda-launch-bench needs the CUDA toolkit and cgo. This file keeps
// `go build ./...` and `go vet ./...` working on every other machine, and
// tells anyone who runs the binary how to get the real one.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "cuda-launch-bench: built without CUDA.")
	fmt.Fprintln(os.Stderr, "Rebuild on a machine with the CUDA toolkit:")
	fmt.Fprintln(os.Stderr, "  go build -tags cuda -o bin/cuda-launch-bench ./cmd/cuda-launch-bench")
	os.Exit(2)
}
