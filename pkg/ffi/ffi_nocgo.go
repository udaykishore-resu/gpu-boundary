//go:build !cgo

// Package ffi measures the cost of the cgo boundary. Built with CGO_ENABLED=0
// there is no boundary to measure, so every entry point reports that plainly
// rather than pretending.
package ffi

import (
	"errors"
	"time"
	"unsafe"
)

// Available reports whether this build has cgo.
const Available = false

// ErrNoCgo is returned by every operation in a CGO_ENABLED=0 build.
var ErrNoCgo = errors.New("ffi: built with CGO_ENABLED=0")

// ErrQueueFull exists so callers compile in both configurations.
var ErrQueueFull = errors.New("ffi: queue full, call Flush")

// Op identifies a queued operation.
type Op int32

// Operation codes, mirrored from the cgo build.
const (
	OpSaxpy Op = 1
	OpScale Op = 2
)

func (o Op) String() string {
	switch o {
	case OpSaxpy:
		return "saxpy"
	case OpScale:
		return "scale"
	default:
		return "op(?)"
	}
}

// Noop does nothing.
func Noop() {}

// Add adds without crossing anything.
func Add(a, b int32) int32 { return a + b }

// Block sleeps in Go, since there is no C to sleep in.
func Block(d time.Duration) { time.Sleep(d) }

// Queue is a stub that satisfies the same interface shape.
type Queue struct {
	Submitted uint64
	Crossings uint64
}

// NewQueue always fails without cgo.
func NewQueue(capacity int) (*Queue, error) { return nil, ErrNoCgo }

func (q *Queue) Len() int                                           { return 0 }
func (q *Queue) Cap() int                                           { return 0 }
func (q *Queue) Enqueue(op Op, alpha float32, x, y []float32) error { return ErrNoCgo }

// EnqueuePtr mirrors the cgo build's signature.
func (q *Queue) EnqueuePtr(op Op, alpha float32, x, y unsafe.Pointer, n int) error {
	return ErrNoCgo
}

func (q *Queue) Flush() error { return ErrNoCgo }
func (q *Queue) SubmitNow(op Op, alpha float32, x, y []float32) error {
	return ErrNoCgo
}
func (q *Queue) Close() error                 { return nil }
func (q *Queue) AmplificationFactor() float64 { return 0 }

// StallConfig mirrors the cgo build's type.
type StallConfig struct {
	Blockers int
	Hold     time.Duration
	Window   time.Duration
	Tick     time.Duration
	InC      bool
}

// StallResult mirrors the cgo build's type.
type StallResult struct {
	Config        StallConfig
	ThreadsBefore int
	ThreadsAfter  int
	Ticks         int
	LateP50       time.Duration
	LateP99       time.Duration
	LateMax       time.Duration
	Completed     int
}

// ThreadGrowth reports how many OS threads the runtime created during the run.
func (r StallResult) ThreadGrowth() int { return r.ThreadsAfter - r.ThreadsBefore }

// MeasureStall returns a zero result without cgo.
func MeasureStall(cfg StallConfig) StallResult { return StallResult{Config: cfg} }
