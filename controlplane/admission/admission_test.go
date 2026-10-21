package admission_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/admission"
)

func TestConcurrencyCeilingIsRespected(t *testing.T) {
	const limit = 4
	l := admission.New(admission.Config{MaxInFlight: limit, MaxQueue: 64})

	var mu sync.Mutex
	inFlight, peak := 0, 0
	var wg sync.WaitGroup

	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rel, err := l.Acquire(context.Background(), time.Time{})
			if err != nil {
				return
			}
			defer rel()

			mu.Lock()
			inFlight++
			if inFlight > peak {
				peak = inFlight
			}
			mu.Unlock()

			time.Sleep(time.Millisecond)

			mu.Lock()
			inFlight--
			mu.Unlock()
		}()
	}
	wg.Wait()

	if peak > limit {
		t.Fatalf("peak concurrency %d exceeded the limit of %d", peak, limit)
	}
	if peak == 0 {
		t.Fatal("nothing was ever admitted")
	}
	if got := l.Stats().InFlight; got != 0 {
		t.Fatalf("InFlight = %d after all releases, want 0", got)
	}
}

func TestQueueFull(t *testing.T) {
	l := admission.New(admission.Config{MaxInFlight: 1, MaxQueue: 1})

	rel, err := l.Acquire(context.Background(), time.Time{})
	if err != nil {
		t.Fatalf("first Acquire: %v", err)
	}
	defer rel()

	// One waiter fills the queue.
	blocked := make(chan struct{})
	go func() {
		r, err := l.Acquire(context.Background(), time.Time{})
		close(blocked)
		if err == nil {
			r()
		}
	}()
	waitForQueued(t, l, 1)

	if _, err := l.Acquire(context.Background(), time.Time{}); !errors.Is(err, admission.ErrQueueFull) {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
}

// TestDeadlineShedding is the behaviour that keeps an overloaded system from
// collapsing: refuse work that cannot finish rather than queueing it.
func TestDeadlineShedding(t *testing.T) {
	l := admission.New(admission.Config{
		MaxInFlight:     1,
		MaxQueue:        16,
		ServiceEstimate: 200 * time.Millisecond,
		ShedMargin:      time.Millisecond,
	})

	rel, err := l.Acquire(context.Background(), time.Time{})
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer rel()

	// A caller with 10ms of budget cannot be served by a 200ms service.
	_, err = l.Acquire(context.Background(), time.Now().Add(10*time.Millisecond))
	if !errors.Is(err, admission.ErrDeadlineUnreachable) {
		t.Fatalf("expected ErrDeadlineUnreachable, got %v", err)
	}

	// A caller with a generous budget is allowed to queue.
	done := make(chan error, 1)
	go func() {
		r, err := l.Acquire(context.Background(), time.Now().Add(5*time.Second))
		if err == nil {
			r()
		}
		done <- err
	}()
	waitForQueued(t, l, 1)

	st := l.Stats()
	if st.Shed != 1 {
		t.Fatalf("Shed = %d, want 1", st.Shed)
	}

	rel()
	if err := <-done; err != nil {
		t.Fatalf("patient caller was refused: %v", err)
	}
}

func TestExpiredDeadlineIsShedImmediately(t *testing.T) {
	l := admission.New(admission.Config{MaxInFlight: 8})
	_, err := l.Acquire(context.Background(), time.Now().Add(-time.Second))
	if !errors.Is(err, admission.ErrDeadlineUnreachable) {
		t.Fatalf("expected ErrDeadlineUnreachable for a past deadline, got %v", err)
	}
}

func TestContextCancellationReleasesWaiter(t *testing.T) {
	l := admission.New(admission.Config{MaxInFlight: 1, MaxQueue: 8})
	rel, err := l.Acquire(context.Background(), time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	defer rel()

	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := l.Acquire(ctx, time.Time{})
		errc <- err
	}()
	waitForQueued(t, l, 1)
	cancel()

	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled waiter never returned")
	}

	if got := l.Stats().Queued; got != 0 {
		t.Fatalf("Queued = %d after cancellation, want 0", got)
	}
}

func TestDoubleReleaseIsSafe(t *testing.T) {
	l := admission.New(admission.Config{MaxInFlight: 2})
	rel, err := l.Acquire(context.Background(), time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	rel()
	rel()
	rel()
	if got := l.Stats().InFlight; got != 0 {
		t.Fatalf("InFlight = %d, want 0", got)
	}
	// The limiter must still hand out both slots.
	a, err := l.Acquire(context.Background(), time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	b, err := l.Acquire(context.Background(), time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	a()
	b()
}

func TestTryAcquire(t *testing.T) {
	l := admission.New(admission.Config{MaxInFlight: 1})
	rel, ok := l.TryAcquire()
	if !ok {
		t.Fatal("first TryAcquire failed")
	}
	if _, ok := l.TryAcquire(); ok {
		t.Fatal("second TryAcquire succeeded past the ceiling")
	}
	rel()
	if _, ok := l.TryAcquire(); !ok {
		t.Fatal("TryAcquire failed after release")
	}
}

func TestClosed(t *testing.T) {
	l := admission.New(admission.Config{MaxInFlight: 1})
	l.Close()
	if _, err := l.Acquire(context.Background(), time.Time{}); !errors.Is(err, admission.ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
}

func waitForQueued(t *testing.T, l *admission.Limiter, want int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if l.Stats().Queued >= want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d queued requests (have %d)", want, l.Stats().Queued)
}
