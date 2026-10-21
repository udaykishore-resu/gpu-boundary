//go:build amd64 && !purego

#include "textflag.h"

// func cpuidex(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuidex(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// func dotAVX2(a, b *float32, n int) float32
//
// Two 8-wide accumulators are carried so that two FMAs can be in flight per
// cycle; a single accumulator would serialise on the 4-cycle FMA latency and
// give up roughly half the throughput.
TEXT ·dotAVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX

	VXORPS Y0, Y0, Y0 // acc0
	VXORPS Y3, Y3, Y3 // acc1
	XORQ   AX, AX     // i

	// Main loop: 16 elements per iteration.
	MOVQ CX, DX
	ANDQ $-16, DX

loop16:
	CMPQ AX, DX
	JGE  tail8prep

	VMOVUPS     (SI)(AX*4), Y1
	VMOVUPS     (DI)(AX*4), Y2
	VFMADD231PS Y2, Y1, Y0

	VMOVUPS     32(SI)(AX*4), Y4
	VMOVUPS     32(DI)(AX*4), Y5
	VFMADD231PS Y5, Y4, Y3

	ADDQ $16, AX
	JMP  loop16

tail8prep:
	VADDPS Y3, Y0, Y0

	// Remainder loop: 8 elements per iteration.
	MOVQ CX, DX
	ANDQ $-8, DX

loop8:
	CMPQ AX, DX
	JGE  reduce

	VMOVUPS     (SI)(AX*4), Y1
	VMOVUPS     (DI)(AX*4), Y2
	VFMADD231PS Y2, Y1, Y0

	ADDQ $8, AX
	JMP  loop8

reduce:
	// Horizontal sum of Y0 into the low lane of X0.
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0

	// Scalar tail for the final (n mod 8) elements.
loop1:
	CMPQ AX, CX
	JGE  done

	VMOVSS (SI)(AX*4), X1
	VMULSS (DI)(AX*4), X1, X1
	VADDSS X1, X0, X0

	INCQ AX
	JMP  loop1

done:
	VZEROUPPER
	MOVSS X0, ret+24(FP)
	RET
