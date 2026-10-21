//go:build amd64 && !purego

package kernel

// The standard library's CPU feature table lives in internal/cpu, which is not
// importable, and golang.org/x/sys/cpu is a dependency this repository does not
// take. Twenty lines of assembly is the honest cost of a zero-dependency
// dispatch table, so we pay it here.

//go:noescape
func cpuidex(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

//go:noescape
func xgetbv() (eax, edx uint32)

// Features describes the vector capabilities of the CPU running this binary.
type Features struct {
	AVX    bool
	AVX2   bool
	FMA    bool
	OSAVX  bool // the OS has enabled saving the YMM state on context switch
	MaxLev uint32
}

var features = detect()

// CPU returns the detected vector features of the host.
func CPU() Features { return features }

func detect() Features {
	var f Features
	maxLev, _, _, _ := cpuidex(0, 0)
	f.MaxLev = maxLev
	if maxLev < 1 {
		return f
	}

	_, _, ecx1, _ := cpuidex(1, 0)
	const (
		bitFMA      = 1 << 12
		bitOSXSAVE  = 1 << 27
		bitAVX      = 1 << 28
		xmmYmmState = 0x6 // XCR0 bits 1 (SSE) and 2 (AVX)
	)
	f.AVX = ecx1&bitAVX != 0
	f.FMA = ecx1&bitFMA != 0

	if ecx1&bitOSXSAVE != 0 {
		eax, _ := xgetbv()
		f.OSAVX = eax&xmmYmmState == xmmYmmState
	}

	if maxLev >= 7 {
		_, ebx7, _, _ := cpuidex(7, 0)
		f.AVX2 = ebx7&(1<<5) != 0
	}
	return f
}

// UseAVX2 reports whether the AVX2+FMA kernel is safe to call. Both the CPU and
// the operating system have to agree: a CPU can advertise AVX2 while the OS
// declines to preserve the wide registers across a context switch, and calling
// the kernel anyway faults.
func UseAVX2() bool {
	return features.AVX && features.AVX2 && features.FMA && features.OSAVX
}

//go:noescape
func dotAVX2(a, b *float32, n int) float32

// DotAVX2 is the hand-written AVX2+FMA kernel. It is the fastest thing a Go
// programmer can ship today without cgo, and writing it required dropping out
// of Go entirely, which is the limitation this file documents.
func DotAVX2(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: DotAVX2 called with mismatched lengths")
	}
	if len(a) == 0 {
		return 0
	}
	return dotAVX2(&a[0], &b[0], len(a))
}

func init() {
	if UseAVX2() {
		Register(Impl{
			Name: "avx2-asm",
			Prio: 50,
			Note: "hand-written Go assembly, 8-wide FMA, 2 accumulators",
			Dot:  DotAVX2,
		})
	}
}
