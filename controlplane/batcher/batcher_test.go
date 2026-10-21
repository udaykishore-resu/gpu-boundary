package batcher_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/udaykishore-resu/gpu-boundary/controlplane/batcher"
)

func echo(ctx context.Context, reqs []int) ([]string, error) {
	out := make([]string, len(reqs))
	for i, r := range reqs {
		out[i] = fmt.Sprintf("v%d", r)
	}
	return out, nil
}

func TestEveryRequestGetsItsOwnResponse(t *testing.T) {
	b, err := batcher.New(batcher.Config[int, string]{
		MaxSize: 8,
		MaxWait: 2 * time.Millisecond,
		Process: echo,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	const n = 200
	var wg sync.WaitGroup
	errs := make([]error, n)
	got := make([]string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got[i], errs[i] = b.Submit(context.Background(), i)
		}(i)
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Fatalf("request %d: %v", i, errs[i])
		}
		if want := fmt.Sprintf("v%d", i); got[i] != want {
			t.Fatalf("request %d got %q want %q: responses are misaligned", i, got[i], want)
		}
	}

	st := b.Stats()
	t.Logf("stats: %+v", st)
	if st.Dispatched != n {
		t.Fatalf("Dispatched = %d, want %d", st.Dispatched, n)
	}
	if st.AvgBatchSize <= 1 {
		t.Fatalf("AvgBatchSize = %v: nothing was actually batched", st.AvgBatchSize)
	}
}

// TestMaxWaitBoundsLatency is the latency half of the batching contract: a lone
// request must not wait indefinitely for company that never arrives.
func TestMaxWaitBoundsLatency(t *testing.T) {
	const wait = 20 * time.Millisecond
	b, err := batcher.New(batcher.Config[int, string]{
		MaxSize: 1000,
		MaxWait: wait,
		Process: echo,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	start := time.Now()
	if _, err := b.Submit(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)

	if elapsed < wait/2 {
		t.Fatalf("returned in %v, before MaxWait of %v elapsed: the timer is not being used", elapsed, wait)
	}
	if elapsed > 10*wait {
		t.Fatalf("returned in %v, far beyond MaxWait of %v", elapsed, wait)
	}
	if st := b.Stats(); st.TimedBatches == 0 {
		t.Fatal("no batch was dispatched by timeout")
	}
}

func TestMaxSizeDispatchesEarly(t *testing.T) {
	const size = 4
	var seen atomic.Int64
	b, err := batcher.New(batcher.Config[int, string]{
		MaxSize: size,
		MaxWait: 10 * time.Second, // effectively never
		Process: func(ctx context.Context, reqs []int) ([]string, error) {
			seen.Add(int64(len(reqs)))
			return echo(ctx, reqs)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	var wg sync.WaitGroup
	for i := 0; i < size; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if _, err := b.Submit(context.Background(), i); err != nil {
				t.Errorf("submit %d: %v", i, err)
			}
		}(i)
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("full batch was not dispatched before MaxWait")
	}

	if st := b.Stats(); st.FullBatches == 0 {
		t.Fatal("FullBatches = 0, want at least 1")
	}
	if got := seen.Load(); got != size {
		t.Fatalf("process saw %d requests, want %d", got, size)
	}
}

func TestProcessErrorReachesEveryCaller(t *testing.T) {
	sentinel := errors.New("backend exploded")
	b, err := batcher.New(batcher.Config[int, string]{
		MaxSize: 4,
		MaxWait: time.Millisecond,
		Process: func(ctx context.Context, reqs []int) ([]string, error) { return nil, sentinel },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if _, err := b.Submit(context.Background(), i); !errors.Is(err, sentinel) {
				t.Errorf("request %d got %v, want the backend error", i, err)
			}
		}(i)
	}
	wg.Wait()
}

func TestShortResponseIsAnError(t *testing.T) {
	b, err := batcher.New(batcher.Config[int, string]{
		MaxSize: 4,
		MaxWait: time.Millisecond,
		Process: func(ctx context.Context, reqs []int) ([]string, error) {
			return []string{"only-one"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := b.Submit(context.Background(), 1); !errors.Is(err, batcher.ErrShortResponse) {
				t.Errorf("got %v, want ErrShortResponse", err)
			}
		}()
	}
	wg.Wait()
}

// TestCancelledCallerDoesNotBlockOthers checks the property that makes a shared
// batch safe: one caller giving up must not take the batch down with it.
func TestCancelledCallerDoesNotBlockOthers(t *testing.T) {
	release := make(chan struct{})
	b, err := batcher.New(batcher.Config[int, string]{
		MaxSize: 2,
		MaxWait: 5 * time.Millisecond,
		Process: func(ctx context.Context, reqs []int) ([]string, error) {
			<-release
			return echo(ctx, reqs)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancelled := make(chan error, 1)
	go func() {
		_, err := b.Submit(ctx, 1)
		cancelled <- err
	}()

	survivor := make(chan error, 1)
	go func() {
		_, err := b.Submit(context.Background(), 2)
		survivor <- err
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case err := <-cancelled:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled caller got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled caller did not return")
	}

	close(release)
	select {
	case err := <-survivor:
		if err != nil {
			t.Fatalf("survivor got %v, want success", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("survivor never completed: one cancellation took down the batch")
	}
}

func TestSubmitAfterCloseFails(t *testing.T) {
	b, err := batcher.New(batcher.Config[int, string]{MaxSize: 2, MaxWait: time.Millisecond, Process: echo})
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Submit(context.Background(), 1); !errors.Is(err, batcher.ErrClosed) {
		t.Fatalf("got %v, want ErrClosed", err)
	}
	if err := b.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestNilProcessRejected(t *testing.T) {
	if _, err := batcher.New(batcher.Config[int, string]{}); err == nil {
		t.Fatal("expected an error when Process is nil")
	}
}

// BenchmarkBatching is the throughput argument. The process function charges a
// fixed cost per batch, mimicking a launch overhead, so a larger MaxSize should
// improve throughput until the queue runs dry.
func BenchmarkBatching(b *testing.B) {
	for _, size := range []int{1, 8, 32} {
		b.Run(fmt.Sprintf("maxsize=%d", size), func(b *testing.B) {
			bat, err := batcher.New(batcher.Config[int, string]{
				MaxSize: size,
				MaxWait: 500 * time.Microsecond,
				Workers: 1,
				Process: func(ctx context.Context, reqs []int) ([]string, error) {
					time.Sleep(200 * time.Microsecond) // fixed launch cost
					return echo(ctx, reqs)
				},
			})
			if err != nil {
				b.Fatal(err)
			}
			defer bat.Close()

			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					if _, err := bat.Submit(context.Background(), 1); err != nil {
						b.Error(err)
						return
					}
				}
			})
			b.StopTimer()
			b.ReportMetric(bat.Stats().AvgBatchSize, "req/batch")
		})
	}
}
