//go:build cgo

// Package cref provides the same dot product in C, reached through cgo.
//
// It exists to answer two questions at once:
//
//  1. How much faster is a C compiler's auto-vectoriser than the Go compiler's
//     (which does not vectorise this loop at all)?
//  2. How much of that win does the cgo call boundary give back on small
//     inputs?
//
// The second question is the interesting one. The C kernel is strictly faster
// per element, and for small vectors it still loses, because the boundary is
// paid once per call regardless of how much work the call does. That crossover
// point is the single most useful number in this repository.
//
// The package lives apart from pkg/kernel so that the kernel package stays
// cgo-free and cross-compilable.
package cref

/*
#cgo CFLAGS: -O3
#cgo !darwin CFLAGS: -march=native

#include <stddef.h>

// dot_c is left to the C compiler's auto-vectoriser on purpose. With -O3 and
// -ffast-math it would vectorise the reduction too; without fast-math it
// respects float associativity, which keeps the numerics comparable to the Go
// implementations.
float dot_c(const float* a, const float* b, size_t n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i+0] * b[i+0];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}
*/
import "C"

import "unsafe"

// Available reports whether the cgo reference kernel was compiled in.
const Available = true

// Dot computes the inner product in C.
//
// The slices are Go memory containing no Go pointers, so they may be passed
// directly to C for the duration of the call without pinning. If C retained
// them past the call, or if we stored them into C memory, runtime.Pinner would
// be mandatory. See pkg/ffi for that case.
func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("cref: Dot called with mismatched lengths")
	}
	if len(a) == 0 {
		return 0
	}
	return float32(C.dot_c(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(len(a)),
	))
}
