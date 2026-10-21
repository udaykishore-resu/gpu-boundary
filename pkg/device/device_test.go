package device_test

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/udaykishore-resu/gpu-boundary/pkg/device"
)

// open returns whichever backend this build has. The same test body then runs
// against the CPU in CI and against a real GPU on a machine built with -tags
// metal or -tags cuda, which is the only way a device abstraction stays honest.
func open(t testing.TB) device.Device {
	t.Helper()
	d, err := device.OpenBest()
	if err != nil {
		t.Fatalf("OpenBest: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	t.Logf("device: %s (registered: %v)", d.Name(), device.Registered())
	return d
}

func TestRoundTrip(t *testing.T) {
	d := open(t)

	const n = 4096
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 0.125
	}

	buf, err := d.Alloc(n)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	defer buf.Free()

	if buf.Len() != n {
		t.Fatalf("Len = %d, want %d", buf.Len(), n)
	}
	if err := d.Upload(buf, src); err != nil {
		t.Fatalf("Upload: %v", err)
	}

	got := make([]float32, n)
	if err := d.Download(got, buf); err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("round trip differs at %d: %v != %v", i, got[i], src[i])
		}
	}
}

func TestSaxpy(t *testing.T) {
	d := open(t)

	const n = 1024
	const alpha = 2.5

	hx := make([]float32, n)
	hy := make([]float32, n)
	want := make([]float32, n)
	for i := range hx {
		hx[i] = float32(i%97) - 48
		hy[i] = float32(i%13) - 6
		want[i] = alpha*hx[i] + hy[i]
	}

	x, err := d.Alloc(n)
	if err != nil {
		t.Fatalf("Alloc x: %v", err)
	}
	defer x.Free()
	y, err := d.Alloc(n)
	if err != nil {
		t.Fatalf("Alloc y: %v", err)
	}
	defer y.Free()

	if err := d.Upload(x, hx); err != nil {
		t.Fatalf("Upload x: %v", err)
	}
	if err := d.Upload(y, hy); err != nil {
		t.Fatalf("Upload y: %v", err)
	}
	if err := d.Saxpy(alpha, x, y); err != nil {
		t.Fatalf("Saxpy: %v", err)
	}
	if err := d.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	got := make([]float32, n)
	if err := d.Download(got, y); err != nil {
		t.Fatalf("Download: %v", err)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("saxpy differs at %d: got %v want %v", i, got[i], want[i])
		}
	}
}

// TestBatchingAmplification is the device-level version of the finding in
// pkg/ffi: many enqueues, one submission.
func TestBatchingAmplification(t *testing.T) {
	d := open(t)

	const n = 256
	const launches = 32

	x, err := d.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	y, err := d.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Free()

	hx := make([]float32, n)
	hy := make([]float32, n)
	for i := range hx {
		hx[i] = 1
	}
	if err := d.Upload(x, hx); err != nil {
		t.Fatal(err)
	}
	if err := d.Upload(y, hy); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < launches; i++ {
		if err := d.Saxpy(1, x, y); err != nil {
			t.Fatalf("Saxpy %d: %v", i, err)
		}
	}
	if err := d.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	got := make([]float32, n)
	if err := d.Download(got, y); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != float32(launches) {
			t.Fatalf("y[%d] = %v after %d launches, want %d", i, got[i], launches, launches)
		}
	}

	st := d.Stats()
	if st.Enqueued < launches {
		t.Fatalf("Enqueued = %d, want at least %d", st.Enqueued, launches)
	}
	if st.Submissions != 1 {
		t.Fatalf("Submissions = %d, want 1: the backend is not batching", st.Submissions)
	}
	if st.Amplification() < float64(launches) {
		t.Fatalf("Amplification = %v, want at least %d", st.Amplification(), launches)
	}
	t.Logf("stats: %+v amplification=%.1f", st, st.Amplification())
}

func TestSizeMismatch(t *testing.T) {
	d := open(t)
	a, err := d.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := d.Alloc(32)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()

	if err := d.Saxpy(1, a, b); !errors.Is(err, device.ErrSizeMismatch) {
		t.Fatalf("expected ErrSizeMismatch, got %v", err)
	}
	if err := d.Upload(a, make([]float32, 8)); !errors.Is(err, device.ErrSizeMismatch) {
		t.Fatalf("expected ErrSizeMismatch on upload, got %v", err)
	}
}

func TestForeignBuffer(t *testing.T) {
	d1 := open(t)
	d2, err := device.OpenBest()
	if err != nil {
		t.Fatalf("second OpenBest: %v", err)
	}
	defer d2.Close()

	b2, err := d2.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer b2.Free()

	if err := d1.Upload(b2, make([]float32, 16)); !errors.Is(err, device.ErrForeignBuffer) {
		t.Fatalf("expected ErrForeignBuffer, got %v", err)
	}
}

func TestDoubleFreeAndClose(t *testing.T) {
	d, err := device.OpenBest()
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Alloc(8)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Free(); err != nil {
		t.Fatalf("Free: %v", err)
	}
	if err := b.Free(); err != nil {
		t.Fatalf("double Free: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("double Close: %v", err)
	}
	if _, err := d.Alloc(8); !errors.Is(err, device.ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
}

func TestUnregisteredBackend(t *testing.T) {
	if _, err := device.Open("nonsense"); !errors.Is(err, device.ErrNotRegistered) {
		t.Fatalf("expected ErrNotRegistered, got %v", err)
	}
}

func BenchmarkSubmission(b *testing.B) {
	d, err := device.OpenBest()
	if err != nil {
		b.Fatal(err)
	}
	defer d.Close()

	const n = 1024
	x, _ := d.Alloc(n)
	y, _ := d.Alloc(n)
	defer x.Free()
	defer y.Free()

	for _, batch := range []int{1, 8, 64} {
		b.Run(benchName(batch), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for k := 0; k < batch; k++ {
					if err := d.Saxpy(1.0001, x, y); err != nil {
						b.Fatal(err)
					}
				}
				if err := d.Sync(); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(batch), "launches/sync")
		})
	}
}

func benchName(batch int) string {
	if batch == 1 {
		return "sync-per-launch"
	}
	return fmt.Sprintf("batch=%d", batch)
}
