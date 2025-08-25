#include "cuda_defines.h"
#include <cuda_runtime.h>
#include <nvcomp/lz4.hpp>
#include <nvcomp/lz4.h>
#include <nvcomp.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Compression context - global to maintain state across checkpoints
typedef struct {
    void* d_compressed_buffer;
    size_t compressed_buffer_size;
    void* h_compressed_buffer;
    size_t h_compressed_buffer_size;
    cudaStream_t stream;
    cudaEvent_t comp_event;
    int compression_level;
    int initialized;
} CompressionContext;

static CompressionContext g_comp_ctx = {0};

extern "C" void CUDA_InitCompression(size_t max_uncompressed_size, int compression_level) {
    if (g_comp_ctx.initialized) return;
    
    // Create CUDA stream for async operations
    CUDA_CALL(cudaStreamCreate(&g_comp_ctx.stream));
    CUDA_CALL(cudaEventCreate(&g_comp_ctx.comp_event));
    
    // Store compression level
    g_comp_ctx.compression_level = compression_level;
    
    // Allocate compressed buffer (2x for safety with gdeflate)
    g_comp_ctx.compressed_buffer_size = max_uncompressed_size * 2;
    CUDA_CALL(cudaMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));
    
    // Allocate host buffer for compressed data
    g_comp_ctx.h_compressed_buffer_size = g_comp_ctx.compressed_buffer_size;
    g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);
    
    g_comp_ctx.initialized = 1;
    printf("nvcomp LZ4 compression initialized (level=%d, buffer=%.2f MB)\n", 
           compression_level, g_comp_ctx.compressed_buffer_size/(1024.0*1024.0));
}

extern "C" size_t CUDA_CompressWavefield(
    float* d_wavefield,          // Device pointer to wavefield
    size_t num_elements,         // Number of float elements
    void** h_compressed_output   // Output: host pointer to compressed data
) {
    if (!g_comp_ctx.initialized) {
        printf("ERROR: Compression not initialized\n");
        return 0;
    }
    
    size_t uncompressed_bytes = num_elements * sizeof(float);
    
    // Prepare nvCOMP LZ4 compression options (we ignore compression_level for now)
    nvcompBatchedLZ4CompressOpts_t comp_opts = nvcompBatchedLZ4CompressDefaultOpts;

    // Choose chunk size (use 64KB unless data is smaller)
    size_t chunk_size = 64 * 1024;
    if (uncompressed_bytes < chunk_size) chunk_size = uncompressed_bytes ? uncompressed_bytes : 64 * 1024;

    // Instantiate manager (bitstream kind NVCOMP_NATIVE so we can query size later)
    nvcomp::LZ4Manager manager(
        chunk_size,
        comp_opts,
        nvcompBatchedLZ4DecompressDefaultOpts,
        g_comp_ctx.stream,
        nvcomp::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE);

    // Configure compression (returns config with required sizes)
    nvcomp::CompressionConfig comp_config = manager.configure_compression(uncompressed_bytes);

    size_t required_bytes = comp_config.max_compressed_buffer_size;
    if (required_bytes > g_comp_ctx.compressed_buffer_size) {
        CUDA_CALL(cudaFree(g_comp_ctx.d_compressed_buffer));
        g_comp_ctx.compressed_buffer_size = required_bytes;
        CUDA_CALL(cudaMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));
        free(g_comp_ctx.h_compressed_buffer);
        g_comp_ctx.h_compressed_buffer_size = g_comp_ctx.compressed_buffer_size;
        g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);
    }

    // Launch async compression
    manager.compress(
        reinterpret_cast<const uint8_t*>(d_wavefield),
        reinterpret_cast<uint8_t*>(g_comp_ctx.d_compressed_buffer),
        comp_config,
        nullptr);

    CUDA_CALL(cudaStreamSynchronize(g_comp_ctx.stream));

    size_t actual_compressed_size = manager.get_compressed_output_size(
        reinterpret_cast<const uint8_t*>(g_comp_ctx.d_compressed_buffer));

    CUDA_CALL(cudaMemcpyAsync(
        g_comp_ctx.h_compressed_buffer,
        g_comp_ctx.d_compressed_buffer,
        actual_compressed_size,
        cudaMemcpyDeviceToHost,
        g_comp_ctx.stream));

    CUDA_CALL(cudaEventRecord(g_comp_ctx.comp_event, g_comp_ctx.stream));
    CUDA_CALL(cudaEventSynchronize(g_comp_ctx.comp_event));

    *h_compressed_output = g_comp_ctx.h_compressed_buffer;
    return actual_compressed_size;
}

// Decompress a compressed buffer back into a provided device buffer (expects same number of floats)
extern "C" int CUDA_DecompressWavefield(
    const void* d_compressed_buffer,
    float* d_output_wavefield,
    size_t expected_num_elements)
{
    if (!g_comp_ctx.initialized) {
        printf("ERROR: Compression not initialized (decompress)\n");
        return 0;
    }

    // We'll create a temporary manager just to parse the header and decompress
    nvcompBatchedLZ4CompressOpts_t comp_opts = nvcompBatchedLZ4CompressDefaultOpts;
    size_t chunk_size = 64 * 1024;
    nvcomp::LZ4Manager manager(
        chunk_size,
        comp_opts,
        nvcompBatchedLZ4DecompressDefaultOpts,
        g_comp_ctx.stream,
        nvcomp::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE);

    // Configure decompression from compressed buffer (NVCOMP_NATIVE automatically reads header)
    auto decomp_config = manager.configure_decompression(
        reinterpret_cast<const uint8_t*>(d_compressed_buffer));

    size_t required_bytes = decomp_config.decomp_data_size;
    if (required_bytes != expected_num_elements * sizeof(float)) {
        printf("Warning: decompressed size (%zu) differs from expected (%zu)\n", required_bytes, expected_num_elements * sizeof(float));
    }

    manager.decompress(
        reinterpret_cast<uint8_t*>(d_output_wavefield),
        reinterpret_cast<const uint8_t*>(d_compressed_buffer),
        decomp_config,
        nullptr);

    CUDA_CALL(cudaStreamSynchronize(g_comp_ctx.stream));
    return 1;
}

// Low-level API path disabled (outdated for nvCOMP 5.x)
extern "C" size_t CUDA_CompressWavefield_LowLevel(
    float* d_wavefield,
    size_t num_elements,
    void** h_compressed_output
) {
    (void)d_wavefield; (void)num_elements; (void)h_compressed_output;
    fprintf(stderr, "CUDA_CompressWavefield_LowLevel disabled: update needed for nvCOMP >=5.x low-level API.\n");
    return 0;
}

extern "C" void CUDA_FinalizeCompression() {
    if (!g_comp_ctx.initialized) return;
    
    CUDA_CALL(cudaFree(g_comp_ctx.d_compressed_buffer));
    free(g_comp_ctx.h_compressed_buffer);
    CUDA_CALL(cudaStreamDestroy(g_comp_ctx.stream));
    CUDA_CALL(cudaEventDestroy(g_comp_ctx.comp_event));
    
    g_comp_ctx.initialized = 0;
}