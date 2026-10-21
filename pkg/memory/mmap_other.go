//go:build !unix

package memory

// mapRegion falls back to the Go heap on platforms without mmap. The arena
// still removes per-iteration allocation, which is most of the benefit, but the
// region is now heap memory: the collector scans the span, the address is only
// stable because a []byte's backing array does not move today, and it must not
// be retained by C.
//
// Extra capacity is requested so that NewArena can align the base upward
// without running past the end.
func mapRegion(size int) ([]byte, bool, error) {
	return make([]byte, size+DefaultAlign), false, nil
}

func unmapRegion(b []byte, offHeap bool) error { return nil }
