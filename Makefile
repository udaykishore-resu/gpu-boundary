# gpu-boundary
#
# Every target here works from a clean clone with no arguments and no
# dependencies beyond the Go toolchain and a C compiler.

GO      ?= go
PKGS    := ./...
BIN     := bin
RUNS    ?= 5
LABEL   ?= $(shell $(GO) env GOOS)-$(shell $(GO) env GOARCH)
PYTHON  ?= python3

.DEFAULT_GOAL := help

## help: list targets
help:
	@echo "gpu-boundary targets:"
	@sed -n 's/^## //p' $(MAKEFILE_LIST) | awk -F': ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## run: build and run the control plane with synthetic load
run: build
	$(BIN)/boundaryd -replicas=4 -load=200

## build: build both commands into bin/
build:
	@mkdir -p $(BIN)
	$(GO) build -o $(BIN)/boundaryd ./cmd/boundaryd
	$(GO) build -o $(BIN)/boundary-bench ./cmd/boundary-bench

## test: vet and test everything the default build reaches, with the race detector
test:
	$(GO) vet $(PKGS)
	$(GO) test -race -count=1 $(PKGS)

## test-short: the same, without the slower measurement tests
test-short:
	$(GO) test -short -count=1 $(PKGS)

## test-nocgo: prove every cgo path has a working pure-Go fallback
test-nocgo:
	CGO_ENABLED=0 $(GO) vet $(PKGS)
	CGO_ENABLED=0 $(GO) test -count=1 $(PKGS)

## test-metal: run the device suite against Metal (macOS, needs Xcode command line tools)
test-metal:
	$(GO) test -tags metal -count=1 -v ./pkg/device/

## test-cuda: run the device suite against CUDA + cuBLAS (needs the CUDA toolkit)
test-cuda:
	$(GO) test -tags cuda -count=1 -v ./pkg/device/

## test-simd: run the kernel suite against Go 1.26 archsimd intrinsics
test-simd:
	GOEXPERIMENT=simd $(GO) test -count=1 -v ./pkg/kernel/

## bench: run every measurement and print the table
bench: build
	$(BIN)/boundary-bench

## bench-quick: the same with reduced iteration counts
bench-quick: build
	$(BIN)/boundary-bench -quick

## bench-json: emit the measurements as JSON, for diffing two runs
bench-json: build
	$(BIN)/boundary-bench -json

## bench-go: run the Go benchmarks rather than the harness
bench-go:
	$(GO) test -run=^$$ -bench=. -benchmem $(PKGS)

## cross: check the module builds for every supported platform
cross:
	@set -e; for t in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64; do \
		os=$${t%/*}; arch=$${t#*/}; \
		printf "%-16s" "$$t"; \
		GOOS=$$os GOARCH=$$arch CGO_ENABLED=0 $(GO) build $(PKGS) && echo ok; \
	done

## asm: check the hand-written assembly assembles for both architectures
asm:
	GOARCH=amd64 $(GO) vet ./pkg/kernel/
	GOOS=linux GOARCH=arm64 $(GO) vet ./pkg/kernel/
	@echo "amd64 and arm64 assembly both pass vet"

## bench-runs: every measurement, RUNS times in fresh processes, saved to results/ (docs/REPRODUCING.md)
bench-runs: build
	$(GO) build -o $(BIN)/bench-runs ./cmd/bench-runs
	$(BIN)/bench-runs -runs $(RUNS) -label $(LABEL) -- $(BIN)/boundary-bench -json

## router-crosscheck: prove the Python baseline makes the same 100k routing decisions as Go
router-crosscheck:
	@mkdir -p $(BIN)
	$(GO) run ./cmd/router-bench -trace > $(BIN)/router-go.trace
	$(PYTHON) bench/python/bench_router.py --trace > $(BIN)/router-py.trace
	@if cmp -s $(BIN)/router-go.trace $(BIN)/router-py.trace; then \
		echo "router-crosscheck: Go and Python traces are identical"; \
	else \
		echo "router-crosscheck: traces differ"; diff $(BIN)/router-go.trace $(BIN)/router-py.trace; exit 1; \
	fi

## bench-router: routing decision cost in Go and in Python, RUNS times each, after the cross-check
bench-router: router-crosscheck
	$(GO) build -o $(BIN)/bench-runs ./cmd/bench-runs
	$(GO) build -o $(BIN)/router-bench ./cmd/router-bench
	$(BIN)/bench-runs -runs $(RUNS) -label $(LABEL)-router-go -- $(BIN)/router-bench -json
	$(BIN)/bench-runs -runs $(RUNS) -label $(LABEL)-router-python -- $(PYTHON) bench/python/bench_router.py --json

## bench-cuda-launch: real GPU launch, batching and pinning costs, RUNS times (needs the CUDA toolkit)
bench-cuda-launch:
	@mkdir -p $(BIN)
	$(GO) build -o $(BIN)/bench-runs ./cmd/bench-runs
	$(GO) build -tags cuda -o $(BIN)/cuda-launch-bench ./cmd/cuda-launch-bench
	$(BIN)/bench-runs -runs $(RUNS) -label $(LABEL)-cuda -- $(BIN)/cuda-launch-bench -json

## bench-paper: tests, then every measurement this machine can run, including CUDA when a GPU is present
bench-paper: test bench-runs bench-router
	@if command -v nvidia-smi >/dev/null 2>&1; then $(MAKE) bench-cuda-launch; \
	else echo "bench-paper: no nvidia-smi, skipping the CUDA row"; fi

## fmt: format everything
fmt:
	$(GO) fmt $(PKGS)
	gofmt -l -w .

## lint: formatting and vet, as CI runs them
lint:
	@test -z "$$(gofmt -l . | tee /dev/stderr)" || (echo "run make fmt"; exit 1)
	$(GO) vet $(PKGS)

## cover: test coverage summary
cover:
	$(GO) test -count=1 -coverprofile=coverage.out $(PKGS)
	$(GO) tool cover -func=coverage.out | tail -1

## clean: remove build output
clean:
	rm -rf $(BIN) coverage.out

.PHONY: help run build test test-short test-nocgo test-metal test-cuda test-simd \
        bench bench-quick bench-json bench-go cross asm fmt lint cover clean \
        bench-runs router-crosscheck bench-router bench-cuda-launch bench-paper
