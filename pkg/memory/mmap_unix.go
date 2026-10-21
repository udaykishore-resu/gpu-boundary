//go:build unix

package memory

import "syscall"

// mapRegion reserves size bytes of anonymous, private memory outside the Go
// heap. The collector never scans it, its address never moves, and it can be
// handed to C and retained indefinitely without runtime.Pinner.
func mapRegion(size int) ([]byte, bool, error) {
	b, err := syscall.Mmap(-1, 0, size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return nil, false, err
	}
	return b, true, nil
}

func unmapRegion(b []byte, offHeap bool) error {
	if !offHeap || b == nil {
		return nil
	}
	return syscall.Munmap(b)
}
