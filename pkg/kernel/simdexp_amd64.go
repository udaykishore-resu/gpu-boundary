//go:build goexperiment.simd && amd64 && !purego

// This file is the Go 1.26 experimental intrinsics path. It is compiled only
// when the toolchain is built or invoked with GOEXPERIMENT=simd:
//
//	GOEXPERIMENT=simd go test ./pkg/kernel/
//
// It is deliberately isolated behind a build tag because the simd/archsimd API
// carries no Go 1 compatibility promise and will change. Everything else in the
// repository builds and runs without it.
//
// It is also the one file in this repository that the default CI job cannot
// compile, which is a fair summary of where Go's intrinsics story stands: real,
// landed, and still gated.

package kernel

import "simd/archsimd"

// dotArchSIMD is the same algorithm as the hand-written AVX2 assembly, written
// in Go. The point of comparing them is not that one is faster, it is that the
// intrinsics version is maintainable by someone who does not read assembly.
func dotArchSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: dotArchSIMD called with mismatched lengths")
	}
	n := len(a)

	acc0 := archsimd.BroadcastFloat32x8(0)
	acc1 := archsimd.BroadcastFloat32x8(0)

	i := 0
	for ; i+16 <= n; i += 16 {
		x0 := archsimd.LoadFloat32x8(a[i : i+8])
		y0 := archsimd.LoadFloat32x8(b[i : i+8])
		acc0 = x0.MulAdd(y0, acc0)

		x1 := archsimd.LoadFloat32x8(a[i+8 : i+16])
		y1 := archsimd.LoadFloat32x8(b[i+8 : i+16])
		acc1 = x1.MulAdd(y1, acc1)
	}
	for ; i+8 <= n; i += 8 {
		x := archsimd.LoadFloat32x8(a[i : i+8])
		y := archsimd.LoadFloat32x8(b[i : i+8])
		acc0 = x.MulAdd(y, acc0)
	}

	acc0 = acc0.Add(acc1)

	// archsimd exposes no horizontal reduction, so the lanes come back out
	// through memory and are summed in scalar Go.
	var lanes [8]float32
	acc0.Store(lanes[:])
	sum := ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))

	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func init() {
	// Reuse the CPUID-based gate from cpu_amd64.go rather than the package's
	// own feature table: one detection path, one place to be wrong.
	if UseAVX2() {
		Register(Impl{
			Name: "archsimd",
			Prio: 60,
			Note: "Go 1.26 simd/archsimd intrinsics, 8-wide FMA, 2 accumulators",
			Dot:  dotArchSIMD,
		})
	}
}
