package kernel_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/udaykishore-resu/gpu-boundary/pkg/kernel"
)

func randSlice(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*2 - 1
	}
	return s
}

// TestImplsAgree checks every registered implementation against a float64
// Kahan reference. Vectorised kernels reassociate the summation, so an exact
// match with the scalar loop is the wrong assertion; agreement to within a
// tolerance scaled by n is the right one.
func TestImplsAgree(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	sizes := []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 1000, 4096, 10007}

	for _, n := range sizes {
		a := randSlice(r, n)
		b := randSlice(r, n)
		want := kernel.DotKahan(a, b)

		for _, impl := range kernel.Impls() {
			got := impl.Dot(a, b)
			// float32 has ~1e-7 relative precision; error grows with the
			// number of accumulated terms.
			tol := 1e-6 * float64(n+1)
			if err := kernel.RelErr(got, want); err > tol && absf(float64(got)-want) > 1e-4 {
				t.Errorf("n=%d impl=%s: got %v want %v (rel err %g > tol %g)",
					n, impl.Name, got, want, err, tol)
			}
		}
	}
}

func absf(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

func TestDotLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on mismatched lengths")
		}
	}()
	kernel.Dot(make([]float32, 4), make([]float32, 5))
}

func TestBestIsRegistered(t *testing.T) {
	best := kernel.Best()
	if best.Dot == nil {
		t.Fatal("Best returned an implementation with no kernel")
	}
	if _, ok := kernel.Lookup(best.Name); !ok {
		t.Fatalf("Best returned %q which Lookup cannot find", best.Name)
	}
	t.Logf("best implementation on this host: %s (%s)", best.Name, best.Note)
	for _, i := range kernel.Impls() {
		t.Logf("  registered: prio=%3d %-12s %s", i.Prio, i.Name, i.Note)
	}
}

// TestAliasedInputs guards the common case of computing the squared norm,
// where both arguments point at the same backing array.
func TestAliasedInputs(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	a := randSlice(r, 1024)
	want := kernel.DotKahan(a, a)
	for _, impl := range kernel.Impls() {
		got := impl.Dot(a, a)
		if kernel.RelErr(got, want) > 1e-4 {
			t.Errorf("impl=%s aliased: got %v want %v", impl.Name, got, want)
		}
	}
}

func BenchmarkDot(b *testing.B) {
	r := rand.New(rand.NewSource(3))
	for _, n := range []int{256, 4096, 65536, 1 << 20} {
		x := randSlice(r, n)
		y := randSlice(r, n)
		for _, impl := range kernel.Impls() {
			b.Run(fmt.Sprintf("n=%d/%s", n, impl.Name), func(b *testing.B) {
				b.SetBytes(int64(n) * 4 * 2)
				var sink float32
				for i := 0; i < b.N; i++ {
					sink = impl.Dot(x, y)
				}
				_ = sink
			})
		}
	}
}
