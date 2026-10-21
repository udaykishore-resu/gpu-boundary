// Package batcher turns a stream of individual requests into batches.
//
// This is the single highest-leverage thing a control plane does for an
// accelerator. Every measurement in pkg/ffi and pkg/device says the same thing
// from a different angle: the fixed cost per submission dominates small work,
// so the way to go faster is to submit less often with more in each submission.
// Batching is that idea applied one layer up, at the request level, where the
// fixed cost is a model launch rather than a cgo call.
//
// The trade is latency for throughput, and it is tunable in exactly two knobs:
// MaxSize caps how much work rides in one batch, MaxWait caps how long the
// first request in a batch waits for company. Neither should be tuned by
// guessing; both should be set from the latency budget and measured.
//
// The implementation is generic over request and response types so the same
// machinery works for inference, embedding, reranking or any other accelerator
// call. It is about a hundred and fifty lines, it is race-free, and it handles
// per-caller cancellation - which is a fair illustration of why this layer
// belongs in Go.
package batcher

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Errors returned by the batcher.
var (
	// ErrClosed means Submit was called after Close.
	ErrClosed = errors.New("batcher: closed")
	// ErrShortResponse means the process function returned fewer responses
	// than it was given requests, which is a bug in the backend rather than a
	// runtime condition.
	ErrShortResponse = errors.New("batcher: process returned fewer responses than requests")
)

// ProcessFunc executes one batch. It must return exactly one response per
// request, in the same order.
type ProcessFunc[Req any, Res any] func(ctx context.Context, reqs []Req) ([]Res, error)

// Config describes a batcher.
type Config[Req any, Res any] struct {
	// MaxSize is the largest batch that will be dispatched.
	MaxSize int
	// MaxWait is how long the oldest queued request waits before the batch is
	// dispatched regardless of size. This is the latency knob.
	MaxWait time.Duration
	// Workers is how many batches may be in flight concurrently. More than
	// one is useful when the backend itself pipelines.
	Workers int
	// Process runs a batch.
	Process ProcessFunc[Req, Res]
	// Now is injectable for tests.
	Now func() time.Time
}

func (c Config[Req, Res]) withDefaults() Config[Req, Res] {
	if c.MaxSize <= 0 {
		c.MaxSize = 16
	}
	if c.MaxWait <= 0 {
		c.MaxWait = 5 * time.Millisecond
	}
	if c.Workers <= 0 {
		c.Workers = 1
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return c
}

// Stats reports batching behaviour. AvgBatchSize is the number that justifies
// the whole package.
type Stats struct {
	Submitted    uint64
	Batches      uint64
	Dispatched   uint64
	Errors       uint64
	Cancelled    uint64
	FullBatches  uint64 // dispatched because MaxSize was reached
	TimedBatches uint64 // dispatched because MaxWait elapsed
	AvgBatchSize float64
	MaxObserved  int
}

type item[Req any, Res any] struct {
	req  Req
	resp chan result[Res]
	ctx  context.Context
}

type result[Res any] struct {
	res Res
	err error
}

// Batcher collects requests and dispatches them in batches.
type Batcher[Req any, Res any] struct {
	cfg Config[Req, Res]

	in     chan item[Req, Res]
	closed chan struct{}
	once   sync.Once
	wg     sync.WaitGroup

	submitted   atomic.Uint64
	batches     atomic.Uint64
	dispatched  atomic.Uint64
	errs        atomic.Uint64
	cancelled   atomic.Uint64
	fullBatches atomic.Uint64
	timeBatches atomic.Uint64
	maxObserved atomic.Int64
}

// New starts a batcher. Close must be called to release its goroutines.
func New[Req any, Res any](cfg Config[Req, Res]) (*Batcher[Req, Res], error) {
	cfg = cfg.withDefaults()
	if cfg.Process == nil {
		return nil, errors.New("batcher: Process must be set")
	}
	b := &Batcher[Req, Res]{
		cfg:    cfg,
		in:     make(chan item[Req, Res], cfg.MaxSize*cfg.Workers),
		closed: make(chan struct{}),
	}
	b.wg.Add(1)
	go b.collect()
	return b, nil
}

// Submit enqueues a request and waits for its response.
//
// Cancelling ctx releases the caller immediately. The request may still be
// executed as part of an in-flight batch, because a batch cannot be partially
// cancelled once it has been handed to the backend; this is stated rather than
// papered over, since a caller that assumes otherwise will double-charge for
// tokens.
func (b *Batcher[Req, Res]) Submit(ctx context.Context, req Req) (Res, error) {
	var zero Res
	it := item[Req, Res]{req: req, resp: make(chan result[Res], 1), ctx: ctx}

	select {
	case <-b.closed:
		return zero, ErrClosed
	default:
	}

	select {
	case b.in <- it:
		b.submitted.Add(1)
	case <-ctx.Done():
		b.cancelled.Add(1)
		return zero, ctx.Err()
	case <-b.closed:
		return zero, ErrClosed
	}

	select {
	case r := <-it.resp:
		return r.res, r.err
	case <-ctx.Done():
		b.cancelled.Add(1)
		return zero, ctx.Err()
	}
}

// collect is the batching loop: accumulate until MaxSize or MaxWait, then
// dispatch. It is the only goroutine that touches the pending slice, so no lock
// is needed and there is no lock to get wrong.
func (b *Batcher[Req, Res]) collect() {
	defer b.wg.Done()

	sem := make(chan struct{}, b.cfg.Workers)
	for i := 0; i < b.cfg.Workers; i++ {
		sem <- struct{}{}
	}

	pending := make([]item[Req, Res], 0, b.cfg.MaxSize)
	var timer *time.Timer
	var timeout <-chan time.Time

	stopTimer := func() {
		if timer != nil {
			timer.Stop()
			timer = nil
			timeout = nil
		}
	}

	dispatch := func(reason *atomic.Uint64) {
		if len(pending) == 0 {
			return
		}
		batch := pending
		pending = make([]item[Req, Res], 0, b.cfg.MaxSize)
		stopTimer()
		reason.Add(1)

		<-sem
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			defer func() { sem <- struct{}{} }()
			b.run(batch)
		}()
	}

	for {
		select {
		case it, ok := <-b.in:
			if !ok {
				dispatch(&b.timeBatches)
				return
			}
			// Drop requests whose caller has already given up, before they
			// cost the accelerator anything.
			if it.ctx != nil && it.ctx.Err() != nil {
				it.resp <- result[Res]{err: it.ctx.Err()}
				continue
			}
			pending = append(pending, it)
			if len(pending) == 1 {
				timer = time.NewTimer(b.cfg.MaxWait)
				timeout = timer.C
			}
			if len(pending) >= b.cfg.MaxSize {
				dispatch(&b.fullBatches)
			}

		case <-timeout:
			dispatch(&b.timeBatches)

		case <-b.closed:
			dispatch(&b.timeBatches)
			// Drain anything already queued so no caller is left hanging.
			for {
				select {
				case it := <-b.in:
					pending = append(pending, it)
					if len(pending) >= b.cfg.MaxSize {
						dispatch(&b.fullBatches)
					}
				default:
					dispatch(&b.timeBatches)
					return
				}
			}
		}
	}
}

func (b *Batcher[Req, Res]) run(batch []item[Req, Res]) {
	b.batches.Add(1)
	b.dispatched.Add(uint64(len(batch)))
	for {
		cur := b.maxObserved.Load()
		if int64(len(batch)) <= cur || b.maxObserved.CompareAndSwap(cur, int64(len(batch))) {
			break
		}
	}

	reqs := make([]Req, len(batch))
	for i := range batch {
		reqs[i] = batch[i].req
	}

	// The batch gets its own context. Deriving it from any one caller's
	// context would let a single cancellation abort work belonging to
	// everyone else in the batch.
	ctx := context.Background()
	resps, err := b.cfg.Process(ctx, reqs)

	if err != nil {
		b.errs.Add(1)
		for i := range batch {
			batch[i].resp <- result[Res]{err: err}
		}
		return
	}
	if len(resps) < len(batch) {
		b.errs.Add(1)
		err := fmt.Errorf("%w: %d responses for %d requests", ErrShortResponse, len(resps), len(batch))
		for i := range batch {
			batch[i].resp <- result[Res]{err: err}
		}
		return
	}
	for i := range batch {
		batch[i].resp <- result[Res]{res: resps[i]}
	}
}

// Stats snapshots batching behaviour.
func (b *Batcher[Req, Res]) Stats() Stats {
	s := Stats{
		Submitted:    b.submitted.Load(),
		Batches:      b.batches.Load(),
		Dispatched:   b.dispatched.Load(),
		Errors:       b.errs.Load(),
		Cancelled:    b.cancelled.Load(),
		FullBatches:  b.fullBatches.Load(),
		TimedBatches: b.timeBatches.Load(),
		MaxObserved:  int(b.maxObserved.Load()),
	}
	if s.Batches > 0 {
		s.AvgBatchSize = float64(s.Dispatched) / float64(s.Batches)
	}
	return s
}

// Close stops the batcher and waits for in-flight batches to finish.
func (b *Batcher[Req, Res]) Close() error {
	b.once.Do(func() { close(b.closed) })
	b.wg.Wait()
	return nil
}
