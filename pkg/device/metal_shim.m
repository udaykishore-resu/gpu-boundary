//go:build darwin && metal

// Objective-C implementation of the Metal backend.
//
// Compiled without ARC, because cgo does not enable it, so object ownership is
// manual: anything obtained from a new/alloc/copy method is owned and must be
// released. This file is short on purpose - the more Objective-C there is, the
// more of the program lives outside anything Go's tooling can analyse.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_shim.h"

#include <string.h>

static id<MTLDevice>               gDevice   = nil;
static id<MTLCommandQueue>         gQueue    = nil;
static id<MTLComputePipelineState> gPipeline = nil;
static id<MTLCommandBuffer>        gCmd      = nil;
static int                         gPending  = 0;
static char                        gError[512];
static char                        gName[256];

static void set_error(NSString *msg) {
    const char *s = msg ? [msg UTF8String] : "unknown error";
    strncpy(gError, s, sizeof(gError) - 1);
    gError[sizeof(gError) - 1] = '\0';
}

static NSString *kernel_source(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void saxpy(device float* y            [[buffer(0)]],\n"
            "                  device const float* x      [[buffer(1)]],\n"
            "                  constant float& alpha      [[buffer(2)]],\n"
            "                  constant uint& n           [[buffer(3)]],\n"
            "                  uint gid [[thread_position_in_grid]]) {\n"
            "    if (gid < n) { y[gid] = alpha * x[gid] + y[gid]; }\n"
            "}\n";
}

int mbx_init(void) {
    if (gPipeline != nil) return 0;
    gError[0] = '\0';

    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (gDevice == nil) {
            set_error(@"MTLCreateSystemDefaultDevice returned nil");
            return 1;
        }
        [gDevice retain];

        strncpy(gName, [[gDevice name] UTF8String], sizeof(gName) - 1);
        gName[sizeof(gName) - 1] = '\0';

        gQueue = [gDevice newCommandQueue];
        if (gQueue == nil) {
            set_error(@"newCommandQueue returned nil");
            return 2;
        }

        NSError *err = nil;
        // Compiling from source at startup avoids a metallib build step, at
        // the cost of a few milliseconds once per process.
        id<MTLLibrary> lib = [gDevice newLibraryWithSource:kernel_source()
                                                   options:nil
                                                     error:&err];
        if (lib == nil) {
            set_error([NSString stringWithFormat:@"newLibraryWithSource: %@",
                                                 err ? [err localizedDescription] : @"nil"]);
            return 3;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"saxpy"];
        if (fn == nil) {
            [lib release];
            set_error(@"kernel function saxpy not found");
            return 4;
        }

        gPipeline = [gDevice newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        [lib release];
        if (gPipeline == nil) {
            set_error([NSString stringWithFormat:@"newComputePipelineState: %@",
                                                 err ? [err localizedDescription] : @"nil"]);
            return 5;
        }
    }
    return 0;
}

const char *mbx_error(void) { return gError; }

const char *mbx_device_name(void) { return gName; }

void *mbx_alloc(size_t bytes) {
    if (gDevice == nil) return NULL;
    id<MTLBuffer> b = [gDevice newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
    return (void *)b;
}

void mbx_free(void *buf) {
    if (buf == NULL) return;
    [(id<MTLBuffer>)buf release];
}

void *mbx_contents(void *buf) {
    if (buf == NULL) return NULL;
    return [(id<MTLBuffer>)buf contents];
}

int mbx_enqueue_saxpy(float alpha, void *x, void *y, uint32_t n) {
    if (gPipeline == nil) {
        set_error(@"metal backend not initialised");
        return 1;
    }
    if (x == NULL || y == NULL || n == 0) {
        set_error(@"invalid saxpy arguments");
        return 2;
    }

    @autoreleasepool {
        if (gCmd == nil) {
            gCmd = [[gQueue commandBuffer] retain];
            if (gCmd == nil) {
                set_error(@"commandBuffer returned nil");
                return 3;
            }
        }

        id<MTLComputeCommandEncoder> enc = [gCmd computeCommandEncoder];
        if (enc == nil) {
            set_error(@"computeCommandEncoder returned nil");
            return 4;
        }

        [enc setComputePipelineState:gPipeline];
        [enc setBuffer:(id<MTLBuffer>)y offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)x offset:0 atIndex:1];
        [enc setBytes:&alpha length:sizeof(float) atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];

        NSUInteger tg = [gPipeline maxTotalThreadsPerThreadgroup];
        if (tg > n) tg = n;
        if (tg == 0) tg = 1;

        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        gPending++;
    }
    return 0;
}

int mbx_commit_wait(void) {
    if (gCmd == nil) return 0;

    [gCmd commit];
    [gCmd waitUntilCompleted];

    MTLCommandBufferStatus st = [gCmd status];
    NSError *err = [gCmd error];
    [gCmd release];
    gCmd = nil;
    gPending = 0;

    if (st != MTLCommandBufferStatusCompleted) {
        set_error([NSString stringWithFormat:@"command buffer status %ld: %@",
                                             (long)st,
                                             err ? [err localizedDescription] : @"no error"]);
        return 1;
    }
    return 0;
}

int mbx_pending(void) { return gPending; }

void mbx_shutdown(void) {
    if (gCmd != nil) {
        [gCmd release];
        gCmd = nil;
    }
    if (gPipeline != nil) {
        [gPipeline release];
        gPipeline = nil;
    }
    if (gQueue != nil) {
        [gQueue release];
        gQueue = nil;
    }
    if (gDevice != nil) {
        [gDevice release];
        gDevice = nil;
    }
    gPending = 0;
}
