// Package admission decides which requests the system agrees to serve.
//
// An inference replica has a hard concurrency ceiling: KV cache is finite and
// oversubscribing it does not degrade gracefully, it thrashes. So the first job
// of a control plane is to say no, early, to work it cannot finish.
//
// Two mechanisms, both boring and both essential:
//
//   - A concurrency limiter with a bounded queue. Unbounded queues do not
//     smooth load, they convert an overload into a latency catastrophe and then
//     an out-of-memory kill.
//   - Deadline-aware shedding. If the estimated wait already exceeds the
//     caller's deadline, the request is refused now rather than occupying a
//     slot to produce a result nobody will read. This single rule is the
//     difference between a system that degrades and one that collapses.
//
// This is the kind of code Go is genuinely excellent at: concurrency,
// cancellation, timers, backpressure. None of the limitations measured
// elsewhere in this repository apply here.
package admission

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// Errors returned by the limiter.
var (
	// ErrQueueFull means the wait queue is at capacity.
	ErrQueueFull = errors.New("admission: queue full")
	// ErrDeadlineUnreachable means the request cannot plausibly finish in
	// time, so it is refused immediately.
	ErrDeadlineUnreachable = errors.New("admission: deadline unreachable, shedding")
	// ErrClosed means the limiter is shutting down.
	ErrClosed = errors.New("admission: closed")
)

// Config describes a limiter.
type Config struct {
	// MaxInFlight is the concurrency ceiling.
	MaxInFlight int
	// MaxQueue is how many requests may wait for a slot.
	MaxQueue int
	// ServiceEstimate is the expected time to serve one request, used to
	// predict queue wait. A rolling measurement replaces it once traffic
	// starts, so the initial value only matters at cold start.
	ServiceEstimate time.Duration
	// ShedMargin is subtracted from the caller's remaining budget before
	// comparing it to the predicted wait, covering network and serialisation
	// time the limiter does not see.
	ShedMargin time.Duration
}

func (c Config) withDefaults() Config {
	if c.MaxInFlight <= 0 {
		c.MaxInFlight = 8
	}
	if c.MaxQueue < 0 {
		c.MaxQueue = 0
	}
	if c.MaxQueue == 0 {
		c.MaxQueue = 4 * c.MaxInFlight
	}
	if c.ServiceEstimate <= 0 {
		c.ServiceEstimate = 50 * time.Millisecond
	}
	if c.ShedMargin <= 0 {
		c.ShedMargin = 5 * time.Millisecond
	}
	return c
}

// Stats is a snapshot of limiter state.
type Stats struct {
	InFlight  int
	Queued    int
	Admitted  uint64
	Shed      uint64
	Rejected  uint64
	Cancelled uint64
	// ServiceEstimate is the current rolling estimate of service time.
	ServiceEstimate time.Duration
	// PredictedWait is what a request arriving now would be told to expect.
	PredictedWait time.Duration
}

// Limiter admits work up to a concurrency ceiling.
type Limiter struct {
	cfg Config

	// slots is the concurrency ceiling, expressed as a buffered channel so
	// that acquisition composes with context cancellation for free. This is
	// the whole implementation of a semaphore in Go and it is a good example
	// of what the language is for.
	slots chan struct{}

	mu       sync.Mutex
	queued   int
	closed   bool
	estimate time.Duration

	inFlight  atomic.Int64
	admitted  atomic.Uint64
	shed      atomic.Uint64
	rejected  atomic.Uint64
	cancelled atomic.Uint64
}

// New returns a limiter.
func New(cfg Config) *Limiter {
	cfg = cfg.withDefaults()
	l := &Limiter{
		cfg:      cfg,
		slots:    make(chan struct{}, cfg.MaxInFlight),
		estimate: cfg.ServiceEstimate,
	}
	for i := 0; i < cfg.MaxInFlight; i++ {
		l.slots <- struct{}{}
	}
	return l
}

// Release returns a slot and records how long the work took, refining the
// service estimate. Calling it twice is a bug; calling it exactly once per
// successful Acquire is mandatory, so callers should defer it.
type Release func()

// Acquire waits for a slot.
//
// It fails fast in three cases, and the distinction between them matters
// operationally: ErrQueueFull means the system is saturated, ErrDeadlineUnreachable
// means this particular caller asked for something impossible, and a context
// error means the caller gave up. Collapsing them into one "unavailable" is how
// dashboards stop being useful.
func (l *Limiter) Acquire(ctx context.Context, deadline time.Time) (Release, error) {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil, ErrClosed
	}
	if l.queued >= l.cfg.MaxQueue {
		l.mu.Unlock()
		l.rejected.Add(1)
		return nil, ErrQueueFull
	}
	wait := l.predictedWaitLocked()
	est := l.estimate
	l.queued++
	l.mu.Unlock()

	dequeue := func() {
		l.mu.Lock()
		l.queued--
		l.mu.Unlock()
	}

	// Shed before queueing if the request cannot finish in time anyway.
	if !deadline.IsZero() {
		budget := time.Until(deadline)
		if budget <= 0 || wait+est+l.cfg.ShedMargin > budget {
			dequeue()
			l.shed.Add(1)
			return nil, ErrDeadlineUnreachable
		}
	}

	select {
	case <-l.slots:
		dequeue()
	case <-ctx.Done():
		dequeue()
		l.cancelled.Add(1)
		return nil, ctx.Err()
	}

	l.inFlight.Add(1)
	l.admitted.Add(1)
	start := time.Now()

	var once sync.Once
	return func() {
		once.Do(func() {
			l.observe(time.Since(start))
			l.inFlight.Add(-1)
			select {
			case l.slots <- struct{}{}:
			default:
				// Only reachable if Release were called more times than
				// Acquire succeeded, which sync.Once already prevents.
			}
		})
	}, nil
}

// TryAcquire takes a slot if one is free right now.
func (l *Limiter) TryAcquire() (Release, bool) {
	select {
	case <-l.slots:
	default:
		l.rejected.Add(1)
		return nil, false
	}
	l.inFlight.Add(1)
	l.admitted.Add(1)
	start := time.Now()
	var once sync.Once
	return func() {
		once.Do(func() {
			l.observe(time.Since(start))
			l.inFlight.Add(-1)
			select {
			case l.slots <- struct{}{}:
			default:
			}
		})
	}, true
}

// observe folds a completed service time into the rolling estimate with an
// exponentially weighted moving average. The weight is deliberately slow: a
// jumpy estimate makes shedding decisions oscillate.
func (l *Limiter) observe(d time.Duration) {
	const alpha = 0.05
	l.mu.Lock()
	defer l.mu.Unlock()
	l.estimate = time.Duration(alpha*float64(d) + (1-alpha)*float64(l.estimate))
}

func (l *Limiter) predictedWaitLocked() time.Duration {
	free := len(l.slots)
	if free > l.queued {
		return 0
	}
	// Requests ahead of this one, divided by the number of servers.
	ahead := l.queued - free
	rounds := (ahead + l.cfg.MaxInFlight - 1) / l.cfg.MaxInFlight
	return time.Duration(rounds) * l.estimate
}

// Stats snapshots the limiter.
func (l *Limiter) Stats() Stats {
	l.mu.Lock()
	queued := l.queued
	est := l.estimate
	wait := l.predictedWaitLocked()
	l.mu.Unlock()
	return Stats{
		InFlight:        int(l.inFlight.Load()),
		Queued:          queued,
		Admitted:        l.admitted.Load(),
		Shed:            l.shed.Load(),
		Rejected:        l.rejected.Load(),
		Cancelled:       l.cancelled.Load(),
		ServiceEstimate: est,
		PredictedWait:   wait,
	}
}

// Close stops admitting new work. In-flight work is unaffected; draining it is
// the caller's job, because only the caller knows how long it is willing to
// wait.
func (l *Limiter) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.closed = true
}
