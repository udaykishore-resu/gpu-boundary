//go:build cgo

// Package ffi measures the cost of the cgo boundary and implements the only
// mitigation that actually works: crossing it less often.
//
// Limitations addressed:
//
//   - Every GPU API is C or C++. Go can only reach one through cgo, and each
//     crossing costs tens of nanoseconds plus a scheduler transition. A model
//     forward pass issues thousands of launches, so the boundary is paid
//     thousands of times per inference.
//   - A blocking C call occupies its OS thread for the whole duration. The Go
//     runtime cannot preempt it, so it hands the P to a new thread and the
//     thread count grows with the number of concurrent blocking calls.
//   - Storing a Go pointer into C memory is illegal unless the pointer is
//     pinned, so any command-buffer design has to manage pinning explicitly.
//
// The mitigation is the Queue type: build a batch of launch descriptors in C
// memory and submit them with one crossing, exactly as CUDA graphs and Metal
// command buffers do for the same reason.
package ffi

/*
#cgo CFLAGS: -O2

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

// noop is the cheapest function that can exist. Timing a call to it measures
// the boundary and nothing else.
static void noop(void) {}

static int32_t add_i32(int32_t a, int32_t b) { return a + b; }

enum { OP_SAXPY = 1, OP_SCALE = 2 };

// bcmd mirrors the shape of a real launch descriptor: an opcode, a problem
// size, a scalar argument and device pointers.
typedef struct {
    int32_t op;
    int32_t n;
    float   alpha;
    float*  x;
    float*  y;
} bcmd;

static void run_one(bcmd* c) {
    int32_t i;
    switch (c->op) {
    case OP_SAXPY:
        for (i = 0; i < c->n; i++) c->y[i] = c->alpha * c->x[i] + c->y[i];
        break;
    case OP_SCALE:
        for (i = 0; i < c->n; i++) c->y[i] = c->alpha * c->y[i];
        break;
    default:
        break;
    }
}

// run_batch executes n descriptors for the price of one boundary crossing.
static void run_batch(bcmd* cs, int32_t n) {
    int32_t k;
    for (k = 0; k < n; k++) run_one(&cs[k]);
}

// block_ns sleeps inside C. The Go scheduler cannot preempt this, which is the
// whole point of exposing it.
static void block_ns(int64_t ns) {
    struct timespec ts;
    ts.tv_sec  = (time_t)(ns / 1000000000LL);
    ts.tv_nsec = (long)(ns % 1000000000LL);
    nanosleep(&ts, NULL);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// Available reports whether this build has cgo, and therefore whether any of
// this package's measurements are meaningful.
const Available = true

// Op identifies a queued operation.
type Op int32

const (
	// OpSaxpy computes y = alpha*x + y.
	OpSaxpy Op = 1
	// OpScale computes y = alpha*y.
	OpScale Op = 2
)

func (o Op) String() string {
	switch o {
	case OpSaxpy:
		return "saxpy"
	case OpScale:
		return "scale"
	default:
		return fmt.Sprintf("op(%d)", int32(o))
	}
}

// Noop crosses into C and immediately returns. Benchmarking it gives the fixed
// cost of the boundary on this machine and toolchain.
func Noop() { C.noop() }

// Add crosses into C to do one integer addition, which is a deliberately
// absurd amount of ceremony for the work involved.
func Add(a, b int32) int32 { return int32(C.add_i32(C.int32_t(a), C.int32_t(b))) }

// Block sleeps for d inside C, holding its OS thread for the duration.
func Block(d time.Duration) { C.block_ns(C.int64_t(d.Nanoseconds())) }

// ErrQueueFull is returned by Enqueue when the queue has no free slot.
var ErrQueueFull = errors.New("ffi: queue full, call Flush")

// Queue accumulates launch descriptors in C memory and submits them in one
// crossing. It is not safe for concurrent use; give each worker its own queue,
// the way a real runtime gives each stream its own command buffer.
type Queue struct {
	cap  int
	n    int
	buf  unsafe.Pointer // C.malloc'd array of C.bcmd
	cmds []C.bcmd       // view over buf, no Go allocation behind it

	// pin keeps every Go slice referenced by a queued descriptor alive and
	// immovable until Flush completes. Without this the runtime is free to
	// move the backing array and C would read freed or stale memory.
	pin runtime.Pinner

	// Submitted counts descriptors executed; Crossings counts cgo calls made
	// to execute them. The ratio is the whole argument for batching.
	Submitted uint64
	Crossings uint64
}

// NewQueue allocates a queue with room for capacity descriptors.
func NewQueue(capacity int) (*Queue, error) {
	if capacity <= 0 {
		return nil, errors.New("ffi: queue capacity must be positive")
	}
	sz := C.size_t(capacity) * C.size_t(unsafe.Sizeof(C.bcmd{}))
	p := C.malloc(sz)
	if p == nil {
		return nil, errors.New("ffi: malloc failed")
	}
	q := &Queue{
		cap:  capacity,
		buf:  unsafe.Pointer(p),
		cmds: unsafe.Slice((*C.bcmd)(unsafe.Pointer(p)), capacity),
	}
	runtime.SetFinalizer(q, func(q *Queue) { q.Close() })
	return q, nil
}

// Len reports how many descriptors are waiting.
func (q *Queue) Len() int { return q.n }

// Cap reports the queue capacity.
func (q *Queue) Cap() int { return q.cap }

// Enqueue records an operation without crossing the boundary.
//
// x may be nil for operations that do not read a second buffer. The slices must
// remain valid until Flush returns; the queue pins them, so the caller does not
// need its own Pinner, but it must not shrink or reallocate them.
func (q *Queue) Enqueue(op Op, alpha float32, x, y []float32) error {
	if q.buf == nil {
		return errors.New("ffi: queue is closed")
	}
	if len(y) == 0 {
		return errors.New("ffi: destination must be non-empty")
	}
	if op == OpSaxpy && len(x) < len(y) {
		return errors.New("ffi: source shorter than destination")
	}
	if q.n == q.cap {
		return ErrQueueFull
	}

	c := &q.cmds[q.n]
	c.op = C.int32_t(op)
	c.n = C.int32_t(len(y))
	c.alpha = C.float(alpha)

	// Pinning is mandatory here and easy to get wrong: we are writing a Go
	// pointer into C memory that outlives this call. runtime.Pinner is the
	// only supported way to do it.
	q.pin.Pin(&y[0])
	c.y = (*C.float)(unsafe.Pointer(&y[0]))
	if len(x) > 0 {
		q.pin.Pin(&x[0])
		c.x = (*C.float)(unsafe.Pointer(&x[0]))
	} else {
		c.x = nil
	}

	q.n++
	return nil
}

// EnqueuePtr records an operation over memory that is not managed by the Go
// heap, and therefore needs no pinning.
//
// This exists because of a measurement, not a hunch. Enqueue has to call
// runtime.Pinner.Pin for every buffer in every descriptor, and pinning costs
// roughly what a cgo crossing costs. Batching Enqueue therefore saves the
// crossing and immediately spends the saving on pinning, and the two cancel
// almost exactly. Feed the queue memory the collector does not manage - see
// pkg/memory.Arena - and the pinning disappears, at which point batching
// delivers the win it is supposed to.
//
// The pointers must remain valid until Flush returns and must not point into
// the Go heap. Passing heap memory here is undefined behaviour that the race
// detector will not catch; that is the price of the fast path.
func (q *Queue) EnqueuePtr(op Op, alpha float32, x, y unsafe.Pointer, n int) error {
	if q.buf == nil {
		return errors.New("ffi: queue is closed")
	}
	if y == nil || n <= 0 {
		return errors.New("ffi: destination must be non-empty")
	}
	if op == OpSaxpy && x == nil {
		return errors.New("ffi: saxpy needs a source")
	}
	if q.n == q.cap {
		return ErrQueueFull
	}

	c := &q.cmds[q.n]
	c.op = C.int32_t(op)
	c.n = C.int32_t(n)
	c.alpha = C.float(alpha)
	c.y = (*C.float)(y)
	c.x = (*C.float)(x)

	q.n++
	return nil
}

// Flush executes every queued descriptor with a single cgo call.
func (q *Queue) Flush() error {
	if q.buf == nil {
		return errors.New("ffi: queue is closed")
	}
	if q.n == 0 {
		return nil
	}
	C.run_batch((*C.bcmd)(q.buf), C.int32_t(q.n))
	q.Submitted += uint64(q.n)
	q.Crossings++
	q.n = 0
	q.pin.Unpin()
	return nil
}

// SubmitNow executes one operation immediately, paying a full boundary crossing
// for it. This is the unbatched path, kept so the benchmark has something
// honest to compare Flush against.
func (q *Queue) SubmitNow(op Op, alpha float32, x, y []float32) error {
	if err := q.Enqueue(op, alpha, x, y); err != nil {
		return err
	}
	return q.Flush()
}

// Close releases the C allocation. The queue must not be used afterwards.
func (q *Queue) Close() error {
	if q.buf == nil {
		return nil
	}
	q.pin.Unpin()
	C.free(q.buf)
	q.buf = nil
	q.cmds = nil
	runtime.SetFinalizer(q, nil)
	return nil
}

// AmplificationFactor reports descriptors executed per boundary crossing. A
// value of 1 means every launch paid the full cgo cost; higher is better.
func (q *Queue) AmplificationFactor() float64 {
	if q.Crossings == 0 {
		return 0
	}
	return float64(q.Submitted) / float64(q.Crossings)
}
