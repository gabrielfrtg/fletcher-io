#include "hip_defines.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_HIPCOMP
#include <hip/hip_runtime.h>

#ifdef COMP_ALGO_ZFP
#include <hipcomp/zfp.h>
#else
// Default to LZ4
#include <hipcomp/lz4.h>
#endif

#include <hipcomp.h>

typedef struct {
    void* d_compressed_buffer;
    size_t compressed_buffer_size;
    void* h_compressed_buffer;
    size_t h_compressed_buffer_size;
    hipStream_t stream;
    hipEvent_t comp_event;
    int compression_level;
    int initialized;
    size_t chunk_size;
#ifdef COMP_ALGO_ZFP
    // ZFP specific fields
    size_t nx, ny, nz;
    double zfp_rate;
#endif
} CompressionContext;

static CompressionContext g_comp_ctx = {0};

#ifdef COMP_ALGO_ZFP

// Default rate for lossy mode if not defined
#ifndef ZFP_RATE
#define ZFP_RATE 24.0
#endif

// Error statistics for ZFP lossy compression validation
typedef struct {
    double max_abs_error;      // Maximum absolute error
    double max_rel_error;      // Maximum relative error
    double rmse;               // Root Mean Square Error
    double psnr;               // Peak Signal-to-Noise Ratio (dB)
    size_t num_samples;        // Number of samples validated
    int validation_enabled;    // Whether to compute error stats
} ZfpErrorStats;

static ZfpErrorStats g_zfp_error_stats = {0};
static float* g_original_copy = NULL;  // Copy of original data for validation
static size_t g_original_size = 0;

// Enable/disable error validation (call before compression)
extern "C" void HIP_SetZfpValidation(int enable) {
    g_zfp_error_stats.validation_enabled = enable;
    if (enable) {
        printf("[ZFP] Error validation ENABLED - will compute error metrics after decompression\n");
    }
}

// Get error statistics after decompression
extern "C" void HIP_GetZfpErrorStats(double* max_abs, double* max_rel, double* rmse, double* psnr) {
    if (max_abs) *max_abs = g_zfp_error_stats.max_abs_error;
    if (max_rel) *max_rel = g_zfp_error_stats.max_rel_error;
    if (rmse) *rmse = g_zfp_error_stats.rmse;
    if (psnr) *psnr = g_zfp_error_stats.psnr;
}

// Print error statistics
extern "C" void HIP_PrintZfpErrorStats() {
    if (g_zfp_error_stats.num_samples == 0) {
        printf("[ZFP] No error statistics available (validation disabled or no decompression done)\n");
        return;
    }
    printf("[ZFP] Compression Error Statistics (rate=%.1f bits/value):\n", g_comp_ctx.zfp_rate);
    printf("      Max Absolute Error: %.6e\n", g_zfp_error_stats.max_abs_error);
    printf("      Max Relative Error: %.6e\n", g_zfp_error_stats.max_rel_error);
    printf("      RMSE:               %.6e\n", g_zfp_error_stats.rmse);
    printf("      PSNR:               %.2f dB\n", g_zfp_error_stats.psnr);
    printf("      Samples validated:  %zu\n", g_zfp_error_stats.num_samples);
}

// ZFP initialization
// NOTE: ZFP GPU backend only supports FIXED_RATE mode!
// Reversible/lossless is NOT supported on CUDA/HIP.
// See: https://zfp.readthedocs.io/en/release1.0.1/execution.html#cuda-limitations
extern "C" void HIP_InitCompression(size_t max_uncompressed_size, int compression_level) {
    if (g_comp_ctx.initialized) return;

    // Create HIP stream for async operations
    HIP_CALL(hipStreamCreate(&g_comp_ctx.stream));
    HIP_CALL(hipEventCreate(&g_comp_ctx.comp_event));

    // Store compression level
    g_comp_ctx.compression_level = compression_level;
    
    // ZFP GPU mode: FIXED_RATE (lossy) - the ONLY mode supported on GPU
    // Rate = bits per value (lower = higher compression, more loss)
    // 32 = minimal loss, 16 = ~2x compression, 8 = ~4x compression
    g_comp_ctx.zfp_rate = ZFP_RATE;
    
    // For fixed-rate, output is predictable: input_size * (rate/32)
    double compression_factor = 32.0 / g_comp_ctx.zfp_rate;
    g_comp_ctx.compressed_buffer_size = (size_t)(max_uncompressed_size / compression_factor) + 4096;
    
    HIP_CALL(hipMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));

    // Allocate host buffer for compressed data
    g_comp_ctx.h_compressed_buffer_size = g_comp_ctx.compressed_buffer_size;
    g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);

    // Check if error validation is enabled via environment variable
    const char* validate_env = getenv("ZFP_VALIDATE");
    if (validate_env && (strcmp(validate_env, "1") == 0 || strcmp(validate_env, "true") == 0)) {
        HIP_SetZfpValidation(1);
    }

    g_comp_ctx.initialized = 1;
    printf("hipCOMP ZFP compression initialized (FIXED_RATE mode, rate=%.1f bits/value, ~%.1fx compression, buffer=%.2f MB)\n",
           g_comp_ctx.zfp_rate, compression_factor, g_comp_ctx.compressed_buffer_size/(1024.0*1024.0));
}

// Set ZFP dimensions (must be called before compression)
extern "C" void HIP_SetZfpDimensions(size_t nx, size_t ny, size_t nz) {
    g_comp_ctx.nx = nx;
    g_comp_ctx.ny = ny;
    g_comp_ctx.nz = nz;
}

extern "C" size_t HIP_CompressWavefield(
    float* d_wavefield,
    size_t num_elements,
    void** h_compressed_output)
{
    if (!g_comp_ctx.initialized) {
        printf("ERROR: Compression not initialized\n");
        return 0;
    }
    
    // If validation enabled, save a copy of original data
    if (g_zfp_error_stats.validation_enabled) {
        size_t data_size = num_elements * sizeof(float);
        if (g_original_size < data_size) {
            if (g_original_copy) free(g_original_copy);
            g_original_copy = (float*)malloc(data_size);
            g_original_size = data_size;
        }
        HIP_CALL(hipMemcpy(g_original_copy, d_wavefield, data_size, hipMemcpyDeviceToHost));
    }
    
    // Setup ZFP options - FIXED_RATE is the ONLY mode supported on GPU
    hipcompZfpOpts opts = hipcompZfpDefaultOpts(
        g_comp_ctx.nx, g_comp_ctx.ny, g_comp_ctx.nz, g_comp_ctx.zfp_rate);
    opts.type = HIPCOMP_ZFP_TYPE_FLOAT;
    opts.dims = 3;
    opts.mode = HIPCOMP_ZFP_MODE_FIXED_RATE;
    
    // Get max compressed size
    size_t max_compressed_size;
    hipcompZfpCompressGetMaxOutputSize(&opts, &max_compressed_size);
    
    // Ensure buffer is large enough
    if (max_compressed_size > g_comp_ctx.compressed_buffer_size) {
        HIP_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
        g_comp_ctx.compressed_buffer_size = max_compressed_size;
        HIP_CALL(hipMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));
    }
    
    // Compress
    size_t compressed_size = 0;
    hipcompStatus_t status = hipcompZfpCompressAsync(
        d_wavefield,
        &opts,
        NULL, 0,  // temp buffer (managed internally)
        g_comp_ctx.d_compressed_buffer,
        &compressed_size,
        g_comp_ctx.stream);
    
    if (status != hipcompSuccess) {
        printf("ERROR: ZFP compression failed with status %d\n", (int)status);
        return 0;
    }
    
    HIP_CALL(hipStreamSynchronize(g_comp_ctx.stream));
    
    // Ensure host buffer is large enough
    if (compressed_size > g_comp_ctx.h_compressed_buffer_size) {
        free(g_comp_ctx.h_compressed_buffer);
        g_comp_ctx.h_compressed_buffer_size = compressed_size;
        g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);
    }
    
    // Copy compressed data to host
    HIP_CALL(hipMemcpy(g_comp_ctx.h_compressed_buffer, g_comp_ctx.d_compressed_buffer,
                       compressed_size, hipMemcpyDeviceToHost));
    
    // If validation enabled, decompress and compute error metrics
    if (g_zfp_error_stats.validation_enabled && g_original_copy) {
        // Allocate temporary buffer for decompressed data
        float* d_decompressed = NULL;
        HIP_CALL(hipMalloc(&d_decompressed, num_elements * sizeof(float)));
        
        // Decompress to temporary buffer
        hipcompStatus_t dec_status = hipcompZfpDecompressAsync(
            g_comp_ctx.d_compressed_buffer,
            compressed_size,
            &opts,
            NULL, 0,
            d_decompressed,
            g_comp_ctx.stream);
        
        if (dec_status == hipcompSuccess) {
            HIP_CALL(hipStreamSynchronize(g_comp_ctx.stream));
            
            // Copy decompressed data to host
            float* h_decompressed = (float*)malloc(num_elements * sizeof(float));
            HIP_CALL(hipMemcpy(h_decompressed, d_decompressed, num_elements * sizeof(float), hipMemcpyDeviceToHost));
            
            // Compute error statistics
            double sum_sq_error = 0.0;
            double max_abs = 0.0;
            double max_rel = 0.0;
            double max_val = g_original_copy[0];
            double min_val = g_original_copy[0];
            
            for (size_t i = 0; i < num_elements; i++) {
                float orig = g_original_copy[i];
                float decomp = h_decompressed[i];
                double error = fabs((double)orig - (double)decomp);
                double rel_error = (fabs(orig) > 1e-10) ? error / fabs(orig) : 0.0;
                
                sum_sq_error += error * error;
                if (error > max_abs) max_abs = error;
                if (rel_error > max_rel) max_rel = rel_error;
                if (orig > max_val) max_val = orig;
                if (orig < min_val) min_val = orig;
            }
            
            double range = max_val - min_val;
            g_zfp_error_stats.max_abs_error = max_abs;
            g_zfp_error_stats.max_rel_error = max_rel;
            g_zfp_error_stats.rmse = sqrt(sum_sq_error / num_elements);
            g_zfp_error_stats.psnr = (range > 1e-10) ? 20.0 * log10(range / g_zfp_error_stats.rmse) : 0.0;
            g_zfp_error_stats.num_samples = num_elements;
            
            free(h_decompressed);
        }
        
        HIP_CALL(hipFree(d_decompressed));
    }
    
    *h_compressed_output = g_comp_ctx.h_compressed_buffer;
    return compressed_size;
}

#else
// Original LZ4 implementation
extern "C" void HIP_InitCompression(size_t max_uncompressed_size, int compression_level) {
    if (g_comp_ctx.initialized) return;

    // Create HIP stream for async operations
    HIP_CALL(hipStreamCreate(&g_comp_ctx.stream));
    HIP_CALL(hipEventCreate(&g_comp_ctx.comp_event));

    // Store compression level
    g_comp_ctx.compression_level = compression_level;

    // Choose chunk size (use 64KB unless data is smaller)
    g_comp_ctx.chunk_size = 64 * 1024;
    if (max_uncompressed_size < g_comp_ctx.chunk_size) {
        g_comp_ctx.chunk_size = max_uncompressed_size ? max_uncompressed_size : 64 * 1024;
    }

    // Allocate compressed buffer (2x for safety)
    g_comp_ctx.compressed_buffer_size = max_uncompressed_size * 2;
    HIP_CALL(hipMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));

    // Allocate host buffer for compressed data
    g_comp_ctx.h_compressed_buffer_size = g_comp_ctx.compressed_buffer_size;
    g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);

    g_comp_ctx.initialized = 1;
    printf("hipCOMP LZ4 compression initialized (level=%d, buffer=%.2f MB, chunk_size=%zu)\n",
           compression_level, g_comp_ctx.compressed_buffer_size/(1024.0*1024.0), g_comp_ctx.chunk_size);
}

extern "C" size_t HIP_CompressWavefield(
    float* d_wavefield,          // Device pointer to wavefield
    size_t num_elements,         // Number of float elements
    void** h_compressed_output   // Output: host pointer to compressed data
) {
    if (!g_comp_ctx.initialized) {
        printf("ERROR: Compression not initialized\n");
        return 0;
    }

    size_t uncompressed_bytes = num_elements * sizeof(float);

    // Calculate number of chunks
    size_t num_chunks = (uncompressed_bytes + g_comp_ctx.chunk_size - 1) / g_comp_ctx.chunk_size;

    // Prepare hipCOMP LZ4 compression options
    hipcompBatchedLZ4Opts_t comp_opts;
    comp_opts.data_type = HIPCOMP_TYPE_CHAR;

    // Query temporary space needed
    size_t temp_bytes;
    hipcompStatus_t status = hipcompBatchedLZ4CompressGetTempSize(
        num_chunks, g_comp_ctx.chunk_size, comp_opts, &temp_bytes);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4CompressGetTempSize failed with status %d\n", (int)status);
        return 0;
    }

    // Allocate temp buffer
    void* d_temp = NULL;
    if (temp_bytes > 0) {
        HIP_CALL(hipMalloc(&d_temp, temp_bytes));
    }

    // Get max compressed chunk size
    size_t max_compressed_chunk_size;
    status = hipcompBatchedLZ4CompressGetMaxOutputChunkSize(
        g_comp_ctx.chunk_size, comp_opts, &max_compressed_chunk_size);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4CompressGetMaxOutputChunkSize failed with status %d\n", (int)status);
        if (d_temp) hipFree(d_temp);
        return 0;
    }

    // Calculate required buffer size
    size_t max_total_compressed = num_chunks * max_compressed_chunk_size;

    // Ensure we have enough space in compressed buffer
    if (max_total_compressed > g_comp_ctx.compressed_buffer_size) {
        HIP_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
        g_comp_ctx.compressed_buffer_size = max_total_compressed;
        HIP_CALL(hipMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));
        free(g_comp_ctx.h_compressed_buffer);
        g_comp_ctx.h_compressed_buffer_size = g_comp_ctx.compressed_buffer_size;
        g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);
    }

    // Setup device arrays for batched compression
    const void** d_in_ptrs;
    void** d_out_ptrs;
    size_t* d_in_bytes;
    size_t* d_out_bytes;

    HIP_CALL(hipMalloc(&d_in_ptrs, num_chunks * sizeof(void*)));
    HIP_CALL(hipMalloc(&d_out_ptrs, num_chunks * sizeof(void*)));
    HIP_CALL(hipMalloc(&d_in_bytes, num_chunks * sizeof(size_t)));
    HIP_CALL(hipMalloc(&d_out_bytes, num_chunks * sizeof(size_t)));

    // Prepare arrays on host
    const void** h_in_ptrs = (const void**)malloc(num_chunks * sizeof(void*));
    void** h_out_ptrs = (void**)malloc(num_chunks * sizeof(void*));
    size_t* h_in_bytes = (size_t*)malloc(num_chunks * sizeof(size_t));

    for (size_t i = 0; i < num_chunks; i++) {
        size_t offset = i * g_comp_ctx.chunk_size;
        size_t remaining = uncompressed_bytes - offset;
        h_in_ptrs[i] = (const void*)((uint8_t*)d_wavefield + offset);
        h_out_ptrs[i] = (void*)((uint8_t*)g_comp_ctx.d_compressed_buffer + i * max_compressed_chunk_size);
        h_in_bytes[i] = (remaining < g_comp_ctx.chunk_size) ? remaining : g_comp_ctx.chunk_size;
    }

    // Copy to device
    HIP_CALL(hipMemcpy(d_in_ptrs, h_in_ptrs, num_chunks * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_out_ptrs, h_out_ptrs, num_chunks * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_in_bytes, h_in_bytes, num_chunks * sizeof(size_t), hipMemcpyHostToDevice));

    // Perform batched compression
    status = hipcompBatchedLZ4CompressAsync(
        d_in_ptrs,
        d_in_bytes,
        g_comp_ctx.chunk_size,
        num_chunks,
        d_temp,
        temp_bytes,
        d_out_ptrs,
        d_out_bytes,
        comp_opts,
        g_comp_ctx.stream);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4CompressAsync failed with status %d\n", (int)status);
        if (d_temp) hipFree(d_temp);
        hipFree(d_in_ptrs);
        hipFree(d_out_ptrs);
        hipFree(d_in_bytes);
        hipFree(d_out_bytes);
        free(h_in_ptrs);
        free(h_out_ptrs);
        free(h_in_bytes);
        return 0;
    }

    HIP_CALL(hipStreamSynchronize(g_comp_ctx.stream));

    // Get actual compressed sizes
    size_t* h_out_bytes = (size_t*)malloc(num_chunks * sizeof(size_t));
    HIP_CALL(hipMemcpy(h_out_bytes, d_out_bytes, num_chunks * sizeof(size_t), hipMemcpyDeviceToHost));

    // Calculate total compressed size
    size_t actual_compressed_size = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        actual_compressed_size += h_out_bytes[i];
    }

    // Format: [num_chunks][chunk_sizes...][compressed_data...]
    size_t metadata_size = sizeof(size_t) + num_chunks * sizeof(size_t);
    size_t total_size = metadata_size + actual_compressed_size;

    if (total_size > g_comp_ctx.h_compressed_buffer_size) {
        free(g_comp_ctx.h_compressed_buffer);
        g_comp_ctx.h_compressed_buffer_size = total_size;
        g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);
    }

    // Write metadata to host buffer
    size_t* metadata = (size_t*)g_comp_ctx.h_compressed_buffer;
    metadata[0] = num_chunks;
    memcpy(&metadata[1], h_out_bytes, num_chunks * sizeof(size_t));

    // Copy compressed chunks contiguously to device temp buffer
    void* d_final_compressed;
    HIP_CALL(hipMalloc(&d_final_compressed, actual_compressed_size));

    size_t offset = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        HIP_CALL(hipMemcpy((uint8_t*)d_final_compressed + offset,
                           h_out_ptrs[i],
                           h_out_bytes[i],
                           hipMemcpyDeviceToDevice));
        offset += h_out_bytes[i];
    }

    // Copy compressed data to host buffer (after metadata)
    HIP_CALL(hipMemcpyAsync(
        (uint8_t*)g_comp_ctx.h_compressed_buffer + metadata_size,
        d_final_compressed,
        actual_compressed_size,
        hipMemcpyDeviceToHost,
        g_comp_ctx.stream));

    HIP_CALL(hipEventRecord(g_comp_ctx.comp_event, g_comp_ctx.stream));
    HIP_CALL(hipEventSynchronize(g_comp_ctx.comp_event));

    // Cleanup
    hipFree(d_final_compressed);
    if (d_temp) hipFree(d_temp);
    hipFree(d_in_ptrs);
    hipFree(d_out_ptrs);
    hipFree(d_in_bytes);
    hipFree(d_out_bytes);
    free(h_in_ptrs);
    free(h_out_ptrs);
    free(h_in_bytes);
    free(h_out_bytes);

    *h_compressed_output = g_comp_ctx.h_compressed_buffer;
    return total_size;
}
#endif // !COMP_ALGO_ZFP (end of LZ4 HIP_CompressWavefield)

#ifdef COMP_ALGO_ZFP
// ZFP Decompress
extern "C" int HIP_DecompressWavefield(
    const void* d_compressed_buffer,
    float* d_output_wavefield,
    size_t expected_num_elements)
{
    if (!g_comp_ctx.initialized) {
        printf("ERROR: Compression not initialized (decompress)\n");
        return 0;
    }
    
    // Get compressed size - for ZFP, estimate from stored buffer size
    size_t compressed_size = g_comp_ctx.h_compressed_buffer_size;
    
    // Setup ZFP options - FIXED_RATE is the ONLY mode supported on GPU
    hipcompZfpOpts opts = hipcompZfpDefaultOpts(
        g_comp_ctx.nx, g_comp_ctx.ny, g_comp_ctx.nz, g_comp_ctx.zfp_rate);
    opts.type = HIPCOMP_ZFP_TYPE_FLOAT;
    opts.dims = 3;
    opts.mode = HIPCOMP_ZFP_MODE_FIXED_RATE;
    
    hipcompStatus_t status = hipcompZfpDecompressAsync(
        d_compressed_buffer,
        compressed_size,
        &opts,
        NULL, 0,  // temp buffer (managed internally)
        d_output_wavefield,
        g_comp_ctx.stream);
    
    if (status != hipcompSuccess) {
        printf("ERROR: ZFP decompression failed with status %d\n", (int)status);
        return 0;
    }
    
    HIP_CALL(hipStreamSynchronize(g_comp_ctx.stream));
    
    // If validation enabled, compute error metrics
    if (g_zfp_error_stats.validation_enabled && g_original_copy) {
        size_t data_size = expected_num_elements * sizeof(float);
        float* h_decompressed = (float*)malloc(data_size);
        HIP_CALL(hipMemcpy(h_decompressed, d_output_wavefield, data_size, hipMemcpyDeviceToHost));
        
        double sum_sq_error = 0.0;
        double max_abs = 0.0;
        double max_rel = 0.0;
        double max_val = 0.0;
        double min_val = 0.0;
        
        for (size_t i = 0; i < expected_num_elements; i++) {
            float orig = g_original_copy[i];
            float decomp = h_decompressed[i];
            double error = fabs((double)orig - (double)decomp);
            double rel_error = (fabs(orig) > 1e-10) ? error / fabs(orig) : 0.0;
            
            sum_sq_error += error * error;
            if (error > max_abs) max_abs = error;
            if (rel_error > max_rel) max_rel = rel_error;
            if (orig > max_val) max_val = orig;
            if (orig < min_val) min_val = orig;
        }
        
        double range = max_val - min_val;
        g_zfp_error_stats.max_abs_error = max_abs;
        g_zfp_error_stats.max_rel_error = max_rel;
        g_zfp_error_stats.rmse = sqrt(sum_sq_error / expected_num_elements);
        g_zfp_error_stats.psnr = (range > 1e-10) ? 20.0 * log10(range / g_zfp_error_stats.rmse) : 0.0;
        g_zfp_error_stats.num_samples = expected_num_elements;
        
        free(h_decompressed);
    }
    
    return 1;
}
#else
// LZ4 Decompress - original implementation
extern "C" int HIP_DecompressWavefield(
    const void* d_compressed_buffer,
    float* d_output_wavefield,
    size_t expected_num_elements)
{
    if (!g_comp_ctx.initialized) {
        printf("ERROR: Compression not initialized (decompress)\n");
        return 0;
    }

    // Read metadata from compressed buffer
    // Format: [num_chunks][chunk_sizes...][compressed_data...]
    size_t num_chunks;
    HIP_CALL(hipMemcpy(&num_chunks, d_compressed_buffer, sizeof(size_t), hipMemcpyDeviceToHost));

    size_t* h_chunk_sizes = (size_t*)malloc(num_chunks * sizeof(size_t));
    HIP_CALL(hipMemcpy(h_chunk_sizes, (uint8_t*)d_compressed_buffer + sizeof(size_t),
                        num_chunks * sizeof(size_t), hipMemcpyDeviceToHost));

    size_t metadata_size = sizeof(size_t) + num_chunks * sizeof(size_t);
    const uint8_t* d_compressed_data = (const uint8_t*)d_compressed_buffer + metadata_size;

    // Setup arrays for batched decompression
    const void** d_in_ptrs;
    void** d_out_ptrs;
    size_t* d_in_bytes;
    size_t* d_out_bytes;
    hipcompStatus_t* d_status;

    HIP_CALL(hipMalloc(&d_in_ptrs, num_chunks * sizeof(void*)));
    HIP_CALL(hipMalloc(&d_out_ptrs, num_chunks * sizeof(void*)));
    HIP_CALL(hipMalloc(&d_in_bytes, num_chunks * sizeof(size_t)));
    HIP_CALL(hipMalloc(&d_out_bytes, num_chunks * sizeof(size_t)));
    HIP_CALL(hipMalloc(&d_status, num_chunks * sizeof(hipcompStatus_t)));

    // Query temp size
    size_t temp_bytes;
    hipcompStatus_t status = hipcompBatchedLZ4DecompressGetTempSize(
        num_chunks, g_comp_ctx.chunk_size, &temp_bytes);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4DecompressGetTempSize failed\n");
        hipFree(d_in_ptrs);
        hipFree(d_out_ptrs);
        hipFree(d_in_bytes);
        hipFree(d_out_bytes);
        hipFree(d_status);
        free(h_chunk_sizes);
        return 0;
    }

    void* d_temp = NULL;
    if (temp_bytes > 0) {
        HIP_CALL(hipMalloc(&d_temp, temp_bytes));
    }

    // Setup pointers and sizes for each chunk
    const void** h_in_ptrs = (const void**)malloc(num_chunks * sizeof(void*));
    void** h_out_ptrs = (void**)malloc(num_chunks * sizeof(void*));
    size_t* h_out_bytes = (size_t*)malloc(num_chunks * sizeof(size_t));

    size_t in_offset = 0;
    size_t out_offset = 0;
    size_t expected_bytes = expected_num_elements * sizeof(float);

    for (size_t i = 0; i < num_chunks; i++) {
        h_in_ptrs[i] = (const void*)(d_compressed_data + in_offset);
        h_out_ptrs[i] = (void*)((uint8_t*)d_output_wavefield + out_offset);

        // Calculate expected uncompressed size for this chunk
        size_t remaining = expected_bytes - out_offset;
        h_out_bytes[i] = (remaining < g_comp_ctx.chunk_size) ? remaining : g_comp_ctx.chunk_size;

        in_offset += h_chunk_sizes[i];
        out_offset += h_out_bytes[i];
    }

    // Copy arrays to device
    HIP_CALL(hipMemcpy(d_in_ptrs, h_in_ptrs, num_chunks * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_out_ptrs, h_out_ptrs, num_chunks * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_in_bytes, h_chunk_sizes, num_chunks * sizeof(size_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_out_bytes, h_out_bytes, num_chunks * sizeof(size_t), hipMemcpyHostToDevice));

    // Perform decompression
    status = hipcompBatchedLZ4DecompressAsync(
        d_in_ptrs,
        d_in_bytes,
        d_out_bytes,
        d_out_bytes,
        num_chunks,
        d_temp,
        temp_bytes,
        d_out_ptrs,
        d_status,
        g_comp_ctx.stream);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4DecompressAsync failed\n");
        if (d_temp) hipFree(d_temp);
        hipFree(d_in_ptrs);
        hipFree(d_out_ptrs);
        hipFree(d_in_bytes);
        hipFree(d_out_bytes);
        hipFree(d_status);
        free(h_in_ptrs);
        free(h_out_ptrs);
        free(h_chunk_sizes);
        free(h_out_bytes);
        return 0;
    }

    HIP_CALL(hipStreamSynchronize(g_comp_ctx.stream));

    // Cleanup
    if (d_temp) hipFree(d_temp);
    hipFree(d_in_ptrs);
    hipFree(d_out_ptrs);
    hipFree(d_in_bytes);
    hipFree(d_out_bytes);
    hipFree(d_status);
    free(h_in_ptrs);
    free(h_out_ptrs);
    free(h_chunk_sizes);
    free(h_out_bytes);

    return 1;
}

#endif // !COMP_ALGO_ZFP (end of LZ4 implementation)

extern "C" void HIP_FinalizeCompression() {
    if (!g_comp_ctx.initialized) return;

#ifdef COMP_ALGO_ZFP
    // Print error statistics if validation was enabled
    if (g_zfp_error_stats.validation_enabled && g_zfp_error_stats.num_samples > 0) {
        HIP_PrintZfpErrorStats();
    }
    // Free validation buffers
    if (g_original_copy) {
        free(g_original_copy);
        g_original_copy = NULL;
        g_original_size = 0;
    }
#endif

    HIP_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
    free(g_comp_ctx.h_compressed_buffer);
    HIP_CALL(hipStreamDestroy(g_comp_ctx.stream));
    HIP_CALL(hipEventDestroy(g_comp_ctx.comp_event));

    g_comp_ctx.initialized = 0;
}

extern "C" void HIP_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                               void** compressed_data, size_t* compressed_size)
{
    extern float* dev_pc;
    const size_t num_elements = ((size_t)sx*sy)*sz;

#ifdef COMP_ALGO_ZFP
    // Set ZFP dimensions for 3D compression
    HIP_SetZfpDimensions(sx, sy, sz);
#endif

   // Use hipCOMP batched LZ4 API wrapper
   *compressed_size = 0;
   *compressed_size = HIP_CompressWavefield(dev_pc, num_elements, compressed_data);

   if (*compressed_size > 0) {
        float compression_ratio = (num_elements * sizeof(float)) / (float)*compressed_size;
        printf("Compressed checkpoint: %.2f MB -> %.2f MB (ratio: %.2fx)\n",
               (num_elements * sizeof(float))/(1024.0*1024.0),
               *compressed_size/(1024.0*1024.0),
               compression_ratio);
    } else {
      printf("Warning: Compression disabled or failed; using uncompressed data\n");
    }
}

extern "C" int HIP_Decompress_to_pc(float* host_compressed, size_t compressed_size,
                    const int sx, const int sy, const int sz)
{
   if (compressed_size == 0) return 0;
#ifdef COMP_ALGO_ZFP
   // Set ZFP dimensions for decompression
   HIP_SetZfpDimensions(sx, sy, sz);
   // Store compressed size for ZFP decompression
   g_comp_ctx.h_compressed_buffer_size = compressed_size;
#endif
   extern float* dev_pc;
   // Upload compressed data to device temp buffer
   void* d_comp = NULL;
   HIP_CALL(hipMalloc(&d_comp, compressed_size));
   HIP_CALL(hipMemcpy(d_comp, host_compressed, compressed_size, hipMemcpyHostToDevice));
   const size_t num_elements = ((size_t)sx*sy)*sz;
   int ok = HIP_DecompressWavefield(d_comp, dev_pc, num_elements);
   HIP_CALL(hipFree(d_comp));
   return ok;
}

extern "C" void HIP_DecompressCheckpointFile(const char* infile,
                           const char* out_header,
                           const char* out_data,
                           int sx, int sy, int sz, int bord, int absorb,
                           float dx, float dy, float dz, float dt_output)
{
   FILE* in = fopen(infile, "rb");
   if (!in) {
      printf("Could not open compressed checkpoint file %s for decompression.\n", infile);
      return;
   }
   FILE* out_bin = fopen(out_data, "wb");
   if (!out_bin) {
      printf("Could not open output data file %s.\n", out_data);
      fclose(in);
      return;
   }
   typedef struct {
      int iteration;
      int nx, ny, nz;
      size_t original_size;
      size_t compressed_size;
      float timestamp;
   } CheckpointHeader;

   int snapshot_count = 0;
   float first_time = 0.0f, second_time = 0.0f;
   while (1) {
      CheckpointHeader header;
      size_t r = fread(&header, sizeof(header), 1, in);
      if (r != 1) break; // EOF
      if (snapshot_count == 0) first_time = header.timestamp; else if (snapshot_count == 1) second_time = header.timestamp;
      if (header.compressed_size == 0 || header.compressed_size > (1ULL<<40)) {
         printf("Invalid compressed_size in header, aborting decompression loop.\n");
         break;
      }
      void* comp_buf = malloc(header.compressed_size);
      if (!comp_buf) { printf("Alloc fail for compressed buffer.\n"); break; }
      if (fread(comp_buf, 1, header.compressed_size, in) != header.compressed_size) {
         printf("Short read on compressed data.\n");
         free(comp_buf);
         break;
      }
      size_t num_floats = header.original_size / sizeof(float);
      float* d_out = NULL;
      HIP_CALL(hipMalloc(&d_out, header.original_size));
      void* d_comp = NULL;
      HIP_CALL(hipMalloc(&d_comp, header.compressed_size));
      HIP_CALL(hipMemcpy(d_comp, comp_buf, header.compressed_size, hipMemcpyHostToDevice));
#ifdef COMP_ALGO_ZFP
      // Set ZFP dimensions from header for this snapshot
      HIP_SetZfpDimensions(header.nx, header.ny, header.nz);
      g_comp_ctx.h_compressed_buffer_size = header.compressed_size;
#endif
      int ok = HIP_DecompressWavefield(d_comp, d_out, num_floats);
      HIP_CALL(hipFree(d_comp));
      if (!ok) {
         printf("Decompression failed for iteration %d.\n", header.iteration);
         HIP_CALL(hipFree(d_out));
         free(comp_buf);
         break;
      }
      float* h_out = (float*)malloc(header.original_size);
      if (!h_out) { printf("Host alloc fail for decompressed output.\n"); }
      else {
         HIP_CALL(hipMemcpy(h_out, d_out, header.original_size, hipMemcpyDeviceToHost));
         fwrite(h_out, 1, header.original_size, out_bin);
         free(h_out);
         snapshot_count++;
      }
      HIP_CALL(hipFree(d_out));
      free(comp_buf);
   }
   fclose(in);
   fclose(out_bin);

   // Write RSF header similar to CloseSliceFile FULL
   FILE* out_hdr = fopen(out_header, "w");
   if (!out_hdr) {
      printf("Could not open header file %s for writing.\n", out_header);
      return;
   }
   const int nx_full = sx;
   const int ny_full = sy;
   const int nz_full = sz;
   float inferred_dt = dt_output;
   if (snapshot_count > 1 && second_time > first_time) {
      inferred_dt = second_time - first_time; // time between snapshots
   }
   fprintf(out_hdr, "in=\"%s\"\n", out_data);
   fprintf(out_hdr, "data_format=\"native_float\"\n");
   fprintf(out_hdr, "esize=%lu\n", sizeof(float));
   fprintf(out_hdr, "n1=%d\n", nx_full);
   fprintf(out_hdr, "n2=%d\n", ny_full);
   fprintf(out_hdr, "n3=%d\n", nz_full);
   fprintf(out_hdr, "n4=%d\n", snapshot_count);
   fprintf(out_hdr, "d1=%f\n", dx);
   fprintf(out_hdr, "d2=%f\n", dy);
   fprintf(out_hdr, "d3=%f\n", dz);
   fprintf(out_hdr, "d4=%f\n", inferred_dt);
   fclose(out_hdr);
   printf("File-level decompression complete: %d snapshots -> %s (%s).\n", snapshot_count, out_data, out_header);
}
#endif
