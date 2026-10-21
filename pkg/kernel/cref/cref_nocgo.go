//go:build !cgo

package cref

// Available reports whether the cgo reference kernel was compiled in.
//
// With CGO_ENABLED=0 the answer is no, and that is worth stating plainly: the
// moment a Go project reaches for C it stops being a single trivially
// cross-compiled static binary. Every consumer of this package has to handle
// its absence, which is the tax the rest of the repository keeps measuring.
const Available = false

// Dot panics. Callers must check Available first.
func Dot(a, b []float32) float32 {
	panic("cref: built with CGO_ENABLED=0; check cref.Available before calling")
}
