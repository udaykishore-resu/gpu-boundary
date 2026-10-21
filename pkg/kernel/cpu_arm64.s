//go:build arm64 && !purego

#include "textflag.h"

// func dotNEON(a, b *float32, n int) float32
//
// NEON is architectural on arm64, so unlike the amd64 path there is no runtime
// feature check to make. Two 4-wide accumulators are carried to keep two FMLAs
// in flight against the ~4 cycle FMLA latency.
//
// Note the horizontal reduction: arm64 has FADDP for this, but the Go
// assembler does not expose a vector float add or a vector pairwise float add,
// so the four lanes have to be moved out one at a time through a general
// purpose register. That awkwardness is itself part of the finding.
TEXT ·dotNEON(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2

	VEOR V0.B16, V0.B16, V0.B16 // acc0
	VEOR V1.B16, V1.B16, V1.B16 // acc1

	MOVD $0, R4      // i
	AND  $-8, R2, R5 // n &^ 7

vloop:
	CMP R5, R4
	BGE vreduce

	VLD1.P 32(R0), [V2.S4, V3.S4]
	VLD1.P 32(R1), [V4.S4, V5.S4]
	VFMLA  V4.S4, V2.S4, V0.S4
	VFMLA  V5.S4, V3.S4, V1.S4

	ADD $8, R4
	B   vloop

vreduce:
	MOVD  $0, R3
	FMOVS R3, F10 // running scalar sum

	VMOV  V0.S[0], R3
	FMOVS R3, F11
	FADDS F11, F10, F10
	VMOV  V0.S[1], R3
	FMOVS R3, F11
	FADDS F11, F10, F10
	VMOV  V0.S[2], R3
	FMOVS R3, F11
	FADDS F11, F10, F10
	VMOV  V0.S[3], R3
	FMOVS R3, F11
	FADDS F11, F10, F10

	VMOV  V1.S[0], R3
	FMOVS R3, F11
	FADDS F11, F10, F10
	VMOV  V1.S[1], R3
	FMOVS R3, F11
	FADDS F11, F10, F10
	VMOV  V1.S[2], R3
	FMOVS R3, F11
	FADDS F11, F10, F10
	VMOV  V1.S[3], R3
	FMOVS R3, F11
	FADDS F11, F10, F10

	// R0 and R1 were advanced by the post-indexed loads and now address
	// element i, so the scalar tail continues from there.
sloop:
	CMP R2, R4
	BGE done

	FMOVS (R0), F11
	FMOVS (R1), F12
	FMULS F12, F11, F11
	FADDS F11, F10, F10

	ADD $4, R0
	ADD $4, R1
	ADD $1, R4
	B   sloop

done:
	FMOVS F10, ret+24(FP)
	RET
