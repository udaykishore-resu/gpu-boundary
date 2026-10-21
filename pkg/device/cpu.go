package device

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/udaykishore-resu/gpu-boundary/pkg/memory"
)

// cpuDevice executes on the host using the best kernel implementation the CPU
// offers, with buffers carved out of an off-heap arena.
//
// It implements the same enqueue/Sync contract as the GPU backends rather than
// executing immediately. That is not ceremony: it keeps the CPU path honest as
// a baseline, and it means a service can be written once against Device and run
// on either without the accelerator path being the only one that batches.
type cpuDevice struct {
	mu     sync.Mutex
	arena  *memory.Arena
	queue  []cpuOp
	stats  Stats
	closed bool
	bufs   map[*cpuBuffer]struct{}
}

type cpuOp struct {
	alpha float32
	x, y  *cpuBuffer
}

type cpuBuffer struct {
	dev  *cpuDevice
	data []float32
	free bool
}

func (b *cpuBuffer) Len() int   { return len(b.data) }
func (b *cpuBuffer) Kind() Kind { return KindCPU }

func (b *cpuBuffer) Free() error {
	if b == nil || b.free {
		return nil
	}
	b.free = true
	b.dev.mu.Lock()
	defer b.dev.mu.Unlock()
	delete(b.dev.bufs, b)
	b.dev.stats.Frees++
	// The arena is bump-allocated, so an individual Free returns nothing to
	// the pool. That is the deal an arena offers and it is stated here so the
	// caller is not surprised: capacity comes back on Close, not per buffer.
	return nil
}

// CPUArenaBytes is the size of the off-heap region a CPU device reserves.
const CPUArenaBytes = 256 << 20

func newCPUDevice() (Device, error) {
	a, err := memory.NewArena(CPUArenaBytes)
	if err != nil {
		return nil, err
	}
	return &cpuDevice{arena: a, bufs: make(map[*cpuBuffer]struct{})}, nil
}

func (d *cpuDevice) Name() string {
	return fmt.Sprintf("host cpu (%s/%s, %d threads)", runtime.GOOS, runtime.GOARCH, runtime.GOMAXPROCS(0))
}

func (d *cpuDevice) Kind() Kind { return KindCPU }

func (d *cpuDevice) Alloc(n int) (Buffer, error) {
	if n <= 0 {
		return nil, fmt.Errorf("device: allocation of %d elements", n)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, ErrClosed
	}
	s, err := d.arena.Float32(n)
	if err != nil {
		return nil, err
	}
	b := &cpuBuffer{dev: d, data: s}
	d.bufs[b] = struct{}{}
	d.stats.Allocs++
	return b, nil
}

func (d *cpuDevice) own(b Buffer) (*cpuBuffer, error) {
	cb, ok := b.(*cpuBuffer)
	if !ok || cb.dev != d {
		return nil, ErrForeignBuffer
	}
	if cb.free {
		return nil, fmt.Errorf("device: buffer already freed")
	}
	return cb, nil
}

func (d *cpuDevice) Upload(dst Buffer, src []float32) error {
	cb, err := d.own(dst)
	if err != nil {
		return err
	}
	if len(src) != len(cb.data) {
		return fmt.Errorf("%w: uploading %d into %d", ErrSizeMismatch, len(src), len(cb.data))
	}
	copy(cb.data, src)
	d.mu.Lock()
	d.stats.Uploads++
	d.stats.BytesUp += uint64(len(src)) * 4
	d.mu.Unlock()
	return nil
}

func (d *cpuDevice) Download(dst []float32, src Buffer) error {
	cb, err := d.own(src)
	if err != nil {
		return err
	}
	if len(dst) != len(cb.data) {
		return fmt.Errorf("%w: downloading %d into %d", ErrSizeMismatch, len(cb.data), len(dst))
	}
	// A download implies the enqueued work must have happened first. Getting
	// this wrong on a real GPU is the classic source of nondeterministic
	// results, so the CPU backend enforces the same ordering.
	if err := d.Sync(); err != nil {
		return err
	}
	copy(dst, cb.data)
	d.mu.Lock()
	d.stats.Downloads++
	d.stats.BytesDown += uint64(len(dst)) * 4
	d.mu.Unlock()
	return nil
}

func (d *cpuDevice) Saxpy(alpha float32, x, y Buffer) error {
	bx, err := d.own(x)
	if err != nil {
		return err
	}
	by, err := d.own(y)
	if err != nil {
		return err
	}
	if len(bx.data) != len(by.data) {
		return fmt.Errorf("%w: saxpy over %d and %d", ErrSizeMismatch, len(bx.data), len(by.data))
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return ErrClosed
	}
	d.queue = append(d.queue, cpuOp{alpha: alpha, x: bx, y: by})
	d.stats.Enqueued++
	return nil
}

func (d *cpuDevice) Sync() error {
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return ErrClosed
	}
	q := d.queue
	d.queue = nil
	if len(q) > 0 {
		d.stats.Submissions++
	}
	d.mu.Unlock()

	for _, op := range q {
		saxpy(op.alpha, op.x.data, op.y.data)
	}
	return nil
}

func (d *cpuDevice) Stats() Stats {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.stats
}

func (d *cpuDevice) Close() error {
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return nil
	}
	d.closed = true
	d.queue = nil
	a := d.arena
	d.arena = nil
	d.bufs = nil
	d.mu.Unlock()
	return a.Close()
}

// saxpy is the host kernel: y = alpha*x + y, four-way unrolled for the same
// instruction-level-parallelism reason as kernel.DotUnrolled.
func saxpy(alpha float32, x, y []float32) {
	n := len(y)
	i := 0
	for ; i+4 <= n; i += 4 {
		xs := x[i : i+4 : i+4]
		ys := y[i : i+4 : i+4]
		ys[0] = alpha*xs[0] + ys[0]
		ys[1] = alpha*xs[1] + ys[1]
		ys[2] = alpha*xs[2] + ys[2]
		ys[3] = alpha*xs[3] + ys[3]
	}
	for ; i < n; i++ {
		y[i] = alpha*x[i] + y[i]
	}
}

func init() {
	Register(KindCPU, 0, newCPUDevice)
}
