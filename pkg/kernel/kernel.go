// Package kernel holds numeric kernels written several different ways so the
// cost of each approach can be measured against the others.
//
// Limitation addressed: the gc compiler optimises for compile speed, not for
// numeric throughput. It does not auto-vectorise floating point reductions, it
// has no stable intrinsics story before Go 1.26's GOEXPERIMENT=simd, and the
// language has no operator overloading to make tensor math readable. The
// working mitigation today is hand-written Go assembly behind a runtime
// dispatch table, which is exactly what this package implements.
//
// Every implementation registers itself at init time with a priority. Best
// returns the highest-priority implementation whose CPU requirements are met on
// the machine actually running the binary, so a single static binary keeps
// working on hardware that lacks the wide instructions.
package kernel

import (
	"math"
	"sort"
	"sync"
)

// DotFunc computes the inner product of two equal-length float32 slices.
type DotFunc func(a, b []float32) float32

// Impl is one registered implementation of the dot kernel.
type Impl struct {
	// Name identifies the implementation in benchmark output.
	Name string
	// Prio orders implementations; the highest available Prio wins.
	Prio int
	// Note explains what the implementation relies on.
	Note string
	// Dot is the kernel itself.
	Dot DotFunc
}

var (
	mu    sync.RWMutex
	impls []Impl
)

// Register adds an implementation to the dispatch table. It is intended to be
// called from init functions in architecture-gated files. Registering a
// duplicate name replaces the earlier entry.
func Register(i Impl) {
	if i.Dot == nil {
		panic("kernel: Register called with nil Dot")
	}
	mu.Lock()
	defer mu.Unlock()
	for idx := range impls {
		if impls[idx].Name == i.Name {
			impls[idx] = i
			return
		}
	}
	impls = append(impls, i)
	sort.SliceStable(impls, func(x, y int) bool { return impls[x].Prio > impls[y].Prio })
}

// Impls returns every registered implementation, highest priority first.
func Impls() []Impl {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]Impl, len(impls))
	copy(out, impls)
	return out
}

// Lookup returns the implementation with the given name.
func Lookup(name string) (Impl, bool) {
	for _, i := range Impls() {
		if i.Name == name {
			return i, true
		}
	}
	return Impl{}, false
}

// Best returns the highest-priority implementation available on this machine.
func Best() Impl {
	mu.RLock()
	defer mu.RUnlock()
	return impls[0]
}

// Dot computes the inner product using the best available implementation.
// It panics if the slices differ in length, matching the standard library's
// convention for programmer error in numeric code.
func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: Dot called with mismatched lengths")
	}
	return Best().Dot(a, b)
}

// DotScalar is the portable baseline: straightforward Go the compiler will not
// vectorise. It is the number every other implementation is measured against.
func DotScalar(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: DotScalar called with mismatched lengths")
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// DotUnrolled is the best a portable Go programmer can do without assembly:
// four independent accumulators to break the serial dependency chain on the
// FPU. It usually buys 2-3x over DotScalar and is still far short of a
// vectorised kernel, which is precisely the gap this repository is about.
func DotUnrolled(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: DotUnrolled called with mismatched lengths")
	}
	var s0, s1, s2, s3 float32
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		// Reslicing to a fixed size hints bounds-check elimination.
		x := a[i : i+4 : i+4]
		y := b[i : i+4 : i+4]
		s0 += x[0] * y[0]
		s1 += x[1] * y[1]
		s2 += x[2] * y[2]
		s3 += x[3] * y[3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// DotKahan is a compensated summation baseline. Vectorised kernels change the
// order of summation, so their results differ from DotScalar in the last bits;
// DotKahan gives the tests a high-accuracy reference to compare against instead
// of pretending float addition is associative.
func DotKahan(a, b []float32) float64 {
	var sum, c float64
	for i := range a {
		y := float64(a[i])*float64(b[i]) - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}
	return sum
}

// RelErr reports the relative error of got against the reference want.
func RelErr(got float32, want float64) float64 {
	if want == 0 {
		return math.Abs(float64(got))
	}
	return math.Abs((float64(got) - want) / want)
}

func init() {
	Register(Impl{
		Name: "scalar",
		Prio: 0,
		Note: "portable Go, one accumulator, no vectorisation",
		Dot:  DotScalar,
	})
	Register(Impl{
		Name: "unrolled",
		Prio: 10,
		Note: "portable Go, 4 accumulators to break the dependency chain",
		Dot:  DotUnrolled,
	})
}
