//go:build arm64 && !purego

package kernel

//go:noescape
func dotNEON(a, b *float32, n int) float32

// DotNEON is the hand-written arm64 kernel. NEON is part of the arm64 baseline,
// so there is no feature gate: if the binary is arm64 the instructions exist.
func DotNEON(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: DotNEON called with mismatched lengths")
	}
	if len(a) == 0 {
		return 0
	}
	return dotNEON(&a[0], &b[0], len(a))
}

// CPU reports the vector features of the host. On arm64 the answer is static.
func CPU() Features { return Features{NEON: true} }

// Features describes the vector capabilities of the CPU running this binary.
type Features struct {
	NEON bool
}

func init() {
	Register(Impl{
		Name: "neon-asm",
		Prio: 50,
		Note: "hand-written Go assembly, 4-wide FMLA, 2 accumulators",
		Dot:  DotNEON,
	})
}
