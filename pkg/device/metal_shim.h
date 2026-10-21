//go:build darwin && metal

// metal_shim.h is the C surface of the Metal backend. Everything above it is
// Go; everything below it is Objective-C. The boundary is drawn here on
// purpose: cgo can call C functions cheaply and predictably, and keeping the
// Objective-C runtime, the autorelease pools and the id<> protocol types on the
// far side of a plain C header is what keeps the Go code readable.

#ifndef GPU_BOUNDARY_METAL_SHIM_H
#define GPU_BOUNDARY_METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>

// mbx_init creates the device, command queue and compute pipeline.
// Returns 0 on success, non-zero on failure; call mbx_error for detail.
int mbx_init(void);

// mbx_error returns the last error message, or an empty string.
const char *mbx_error(void);

// mbx_device_name returns the GPU name, valid until mbx_shutdown.
const char *mbx_device_name(void);

// mbx_alloc allocates a shared-storage MTLBuffer of the given size.
void *mbx_alloc(size_t bytes);

// mbx_free releases a buffer returned by mbx_alloc.
void mbx_free(void *buf);

// mbx_contents returns the host-visible pointer for a shared buffer. On Apple
// silicon this is the same physical memory the GPU reads, so upload and
// download are memcpy rather than DMA.
void *mbx_contents(void *buf);

// mbx_enqueue_saxpy encodes y = alpha*x + y into the current command buffer
// without submitting it. Returns 0 on success.
int mbx_enqueue_saxpy(float alpha, void *x, void *y, uint32_t n);

// mbx_commit_wait submits the current command buffer and blocks until the GPU
// has finished it. Returns 0 on success. This is the only call that talks to
// the driver, which is what makes batching measurable.
int mbx_commit_wait(void);

// mbx_pending reports how many kernels are encoded but not yet submitted.
int mbx_pending(void);

// mbx_shutdown releases the pipeline, queue and device.
void mbx_shutdown(void);

#endif // GPU_BOUNDARY_METAL_SHIM_H
