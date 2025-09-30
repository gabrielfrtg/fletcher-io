#include "cuda_defines.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_HIPCOMP
#include <hip/hip_runtime.h>
#include <hipcomp/lz4.h>
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
} CompressionContext;

static CompressionContext g_comp_ctx = {0};

extern "C" void CUDA_InitCompression(size_t max_uncompressed_size, int compression_level) {
    if (g_comp_ctx.initialized) return;

    // Create HIP stream for async operations
    CUDA_CALL(hipStreamCreate(&g_comp_ctx.stream));
    CUDA_CALL(hipEventCreate(&g_comp_ctx.comp_event));

    // Store compression level
    g_comp_ctx.compression_level = compression_level;

    // Choose chunk size (use 64KB unless data is smaller)
    g_comp_ctx.chunk_size = 64 * 1024;
    if (max_uncompressed_size < g_comp_ctx.chunk_size) {
        g_comp_ctx.chunk_size = max_uncompressed_size ? max_uncompressed_size : 64 * 1024;
    }

    // Allocate compressed buffer (2x for safety)
    g_comp_ctx.compressed_buffer_size = max_uncompressed_size * 2;
    CUDA_CALL(hipMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));

    // Allocate host buffer for compressed data
    g_comp_ctx.h_compressed_buffer_size = g_comp_ctx.compressed_buffer_size;
    g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);

    g_comp_ctx.initialized = 1;
    printf("hipCOMP LZ4 compression initialized (level=%d, buffer=%.2f MB, chunk_size=%zu)\n",
           compression_level, g_comp_ctx.compressed_buffer_size/(1024.0*1024.0), g_comp_ctx.chunk_size);
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

    // Calculate number of chunks
    size_t num_chunks = (uncompressed_bytes + g_comp_ctx.chunk_size - 1) / g_comp_ctx.chunk_size;

    // Declare all variables at the beginning to avoid goto issues
    size_t* d_uncompressed_bytes = nullptr;
    size_t* d_uncompressed_offsets = nullptr;
    size_t* d_compressed_bytes = nullptr;
    size_t* d_compressed_offsets = nullptr;
    size_t* h_uncompressed_bytes = nullptr;
    size_t* h_uncompressed_offsets = nullptr;
    size_t* h_compressed_offsets = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    size_t max_compressed_chunk_size = 0;
    size_t total_compressed_max = 0;
    size_t actual_compressed_size = 0;
    size_t total_size = 0;
    hipcompStatus_t status;
    hipcompBatchedLZ4Opts_t comp_opts;

    typedef struct {
        size_t num_chunks;
        size_t chunk_size;
        size_t original_size;
    } CompressionHeader;
    CompressionHeader header;

    // Allocate device arrays for chunk information
    CUDA_CALL(hipMalloc(&d_uncompressed_bytes, num_chunks * sizeof(size_t)));
    CUDA_CALL(hipMalloc(&d_uncompressed_offsets, num_chunks * sizeof(size_t)));
    CUDA_CALL(hipMalloc(&d_compressed_bytes, num_chunks * sizeof(size_t)));
    CUDA_CALL(hipMalloc(&d_compressed_offsets, num_chunks * sizeof(size_t)));

    // Prepare chunk sizes on host, then copy to device
    h_uncompressed_bytes = (size_t*)malloc(num_chunks * sizeof(size_t));
    h_uncompressed_offsets = (size_t*)malloc(num_chunks * sizeof(size_t));
    h_compressed_offsets = (size_t*)malloc(num_chunks * sizeof(size_t));

    size_t offset = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        h_uncompressed_offsets[i] = offset;
        size_t remaining = uncompressed_bytes - offset;
        h_uncompressed_bytes[i] = (remaining < g_comp_ctx.chunk_size) ? remaining : g_comp_ctx.chunk_size;
        offset += h_uncompressed_bytes[i];
    }

    CUDA_CALL(hipMemcpy(d_uncompressed_bytes, h_uncompressed_bytes,
                        num_chunks * sizeof(size_t), hipMemcpyHostToDevice));
    CUDA_CALL(hipMemcpy(d_uncompressed_offsets, h_uncompressed_offsets,
                        num_chunks * sizeof(size_t), hipMemcpyHostToDevice));

    // Prepare hipCOMP LZ4 compression options
    comp_opts.data_type = HIPCOMP_TYPE_CHAR;

    // Query temporary space needed
    status = hipcompBatchedLZ4CompressGetTempSize(
        num_chunks, g_comp_ctx.chunk_size, comp_opts, &temp_bytes);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4CompressGetTempSize failed\n");
        goto cleanup;
    }

    if (temp_bytes > 0) {
        CUDA_CALL(hipMalloc(&d_temp, temp_bytes));
    }

    // Get max compressed chunk size
    status = hipcompBatchedLZ4CompressGetMaxOutputChunkSize(
        g_comp_ctx.chunk_size, comp_opts, &max_compressed_chunk_size);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4CompressGetMaxOutputChunkSize failed\n");
        if (d_temp) CUDA_CALL(hipFree(d_temp));
        goto cleanup;
    }

    // Calculate compressed offsets (worst case)
    total_compressed_max = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        h_compressed_offsets[i] = total_compressed_max;
        total_compressed_max += max_compressed_chunk_size;
    }

    // Ensure we have enough space in compressed buffer
    if (total_compressed_max > g_comp_ctx.compressed_buffer_size) {
        CUDA_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
        g_comp_ctx.compressed_buffer_size = total_compressed_max;
        CUDA_CALL(hipMalloc(&g_comp_ctx.d_compressed_buffer, g_comp_ctx.compressed_buffer_size));
        free(g_comp_ctx.h_compressed_buffer);
        g_comp_ctx.h_compressed_buffer_size = g_comp_ctx.compressed_buffer_size;
        g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);
    }

    CUDA_CALL(hipMemcpy(d_compressed_offsets, h_compressed_offsets,
                        num_chunks * sizeof(size_t), hipMemcpyHostToDevice));

    // Perform batched compression
    status = hipcompBatchedLZ4CompressAsync(
        (const void* const*)&d_wavefield,
        d_uncompressed_bytes,
        g_comp_ctx.chunk_size,
        num_chunks,
        d_temp,
        temp_bytes,
        (void* const*)&g_comp_ctx.d_compressed_buffer,
        d_compressed_bytes,
        comp_opts,
        g_comp_ctx.stream);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4CompressAsync failed with status %d\n", (int)status);
        if (d_temp) CUDA_CALL(hipFree(d_temp));
        goto cleanup;
    }

    CUDA_CALL(hipStreamSynchronize(g_comp_ctx.stream));

    // Get actual compressed sizes from device
    CUDA_CALL(hipMemcpy(h_compressed_offsets, d_compressed_bytes,
                        num_chunks * sizeof(size_t), hipMemcpyDeviceToHost));

    // Calculate total compressed size
    actual_compressed_size = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        actual_compressed_size += h_compressed_offsets[i];
    }

    // Add header with metadata (num_chunks, chunk_size, original_size)
    header.num_chunks = num_chunks;
    header.chunk_size = g_comp_ctx.chunk_size;
    header.original_size = uncompressed_bytes;

    // Copy header + compressed data to host buffer
    total_size = sizeof(CompressionHeader) + actual_compressed_size;
    if (total_size > g_comp_ctx.h_compressed_buffer_size) {
        free(g_comp_ctx.h_compressed_buffer);
        g_comp_ctx.h_compressed_buffer_size = total_size;
        g_comp_ctx.h_compressed_buffer = malloc(g_comp_ctx.h_compressed_buffer_size);
    }

    memcpy(g_comp_ctx.h_compressed_buffer, &header, sizeof(CompressionHeader));
    CUDA_CALL(hipMemcpyAsync(
        (char*)g_comp_ctx.h_compressed_buffer + sizeof(CompressionHeader),
        g_comp_ctx.d_compressed_buffer,
        actual_compressed_size,
        hipMemcpyDeviceToHost,
        g_comp_ctx.stream));

    CUDA_CALL(hipEventRecord(g_comp_ctx.comp_event, g_comp_ctx.stream));
    CUDA_CALL(hipEventSynchronize(g_comp_ctx.comp_event));

    if (d_temp) CUDA_CALL(hipFree(d_temp));

cleanup:
    CUDA_CALL(hipFree(d_uncompressed_bytes));
    CUDA_CALL(hipFree(d_uncompressed_offsets));
    CUDA_CALL(hipFree(d_compressed_bytes));
    CUDA_CALL(hipFree(d_compressed_offsets));
    free(h_uncompressed_bytes);
    free(h_uncompressed_offsets);
    free(h_compressed_offsets);

    *h_compressed_output = g_comp_ctx.h_compressed_buffer;
    return total_size;
}

// Decompress a compressed buffer back into a provided device buffer
extern "C" int CUDA_DecompressWavefield(
    const void* d_compressed_buffer,
    float* d_output_wavefield,
    size_t expected_num_elements)
{
    if (!g_comp_ctx.initialized) {
        printf("ERROR: Compression not initialized (decompress)\n");
        return 0;
    }

    // Declare all variables at the beginning to avoid goto issues
    typedef struct {
        size_t num_chunks;
        size_t chunk_size;
        size_t original_size;
    } CompressionHeader;

    CompressionHeader header;
    size_t num_chunks;
    size_t chunk_size;
    const uint8_t* d_compressed_data;
    size_t* d_compressed_bytes = nullptr;
    size_t* d_compressed_offsets = nullptr;
    size_t* d_uncompressed_bytes = nullptr;
    size_t* d_uncompressed_offsets = nullptr;
    hipcompStatus_t* d_status = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    hipcompStatus_t status;
    hipcompBatchedLZ4Opts_t decomp_opts;

    // Read header from compressed buffer (on device)
    CUDA_CALL(hipMemcpy(&header, d_compressed_buffer, sizeof(CompressionHeader), hipMemcpyDeviceToHost));

    if (header.original_size != expected_num_elements * sizeof(float)) {
        printf("Warning: decompressed size (%zu) differs from expected (%zu)\n",
               header.original_size, expected_num_elements * sizeof(float));
    }

    num_chunks = header.num_chunks;
    chunk_size = header.chunk_size;
    d_compressed_data = (const uint8_t*)d_compressed_buffer + sizeof(CompressionHeader);

    // Allocate device arrays for chunk information
    CUDA_CALL(hipMalloc(&d_compressed_bytes, num_chunks * sizeof(size_t)));
    CUDA_CALL(hipMalloc(&d_compressed_offsets, num_chunks * sizeof(size_t)));
    CUDA_CALL(hipMalloc(&d_uncompressed_bytes, num_chunks * sizeof(size_t)));
    CUDA_CALL(hipMalloc(&d_uncompressed_offsets, num_chunks * sizeof(size_t)));
    CUDA_CALL(hipMalloc(&d_status, num_chunks * sizeof(hipcompStatus_t)));

    // Get decompressed sizes for all chunks
    decomp_opts.data_type = HIPCOMP_TYPE_CHAR;

    status = hipcompBatchedLZ4GetDecompressSizeAsync(
        (const void* const*)&d_compressed_data,
        d_compressed_bytes,
        d_uncompressed_bytes,
        num_chunks,
        g_comp_ctx.stream);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4GetDecompressSizeAsync failed\n");
        goto decomp_cleanup;
    }

    CUDA_CALL(hipStreamSynchronize(g_comp_ctx.stream));

    // Query temporary space needed
    status = hipcompBatchedLZ4DecompressGetTempSize(num_chunks, chunk_size, &temp_bytes);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4DecompressGetTempSize failed\n");
        goto decomp_cleanup;
    }

    if (temp_bytes > 0) {
        CUDA_CALL(hipMalloc(&d_temp, temp_bytes));
    }

    // Perform batched decompression
    status = hipcompBatchedLZ4DecompressAsync(
        (const void* const*)&d_compressed_data,
        d_compressed_bytes,
        d_uncompressed_bytes,
        d_uncompressed_bytes,
        num_chunks,
        d_temp,
        temp_bytes,
        (void* const*)&d_output_wavefield,
        d_status,
        g_comp_ctx.stream);

    if (status != hipcompSuccess) {
        printf("ERROR: hipcompBatchedLZ4DecompressAsync failed\n");
        if (d_temp) CUDA_CALL(hipFree(d_temp));
        goto decomp_cleanup;
    }

    CUDA_CALL(hipStreamSynchronize(g_comp_ctx.stream));

    if (d_temp) CUDA_CALL(hipFree(d_temp));

decomp_cleanup:
    CUDA_CALL(hipFree(d_compressed_bytes));
    CUDA_CALL(hipFree(d_compressed_offsets));
    CUDA_CALL(hipFree(d_uncompressed_bytes));
    CUDA_CALL(hipFree(d_uncompressed_offsets));
    CUDA_CALL(hipFree(d_status));

    return (status == hipcompSuccess) ? 1 : 0;
}

extern "C" void CUDA_FinalizeCompression() {
    if (!g_comp_ctx.initialized) return;

    CUDA_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
    free(g_comp_ctx.h_compressed_buffer);
    CUDA_CALL(hipStreamDestroy(g_comp_ctx.stream));
    CUDA_CALL(hipEventDestroy(g_comp_ctx.comp_event));

    g_comp_ctx.initialized = 0;
}

extern "C" void CUDA_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                               void** compressed_data, size_t* compressed_size)
{
    extern float* dev_pc;
    const size_t num_elements = ((size_t)sx*sy)*sz;

   // Use hipCOMP batched LZ4 API wrapper
   *compressed_size = 0;
   *compressed_size = CUDA_CompressWavefield(dev_pc, num_elements, compressed_data);

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

extern "C" int CUDA_Decompress_to_pc(float* host_compressed, size_t compressed_size,
                    const int sx, const int sy, const int sz)
{
   if (compressed_size == 0) return 0;
   extern float* dev_pc;
   // Upload compressed data to device temp buffer
   void* d_comp = nullptr;
   CUDA_CALL(hipMalloc(&d_comp, compressed_size));
   CUDA_CALL(hipMemcpy(d_comp, host_compressed, compressed_size, hipMemcpyHostToDevice));
   const size_t num_elements = ((size_t)sx*sy)*sz;
   int ok = CUDA_DecompressWavefield(d_comp, dev_pc, num_elements);
   CUDA_CALL(hipFree(d_comp));
   return ok;
}

extern "C" void CUDA_DecompressCheckpointFile(const char* infile,
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
      CUDA_CALL(hipMalloc(&d_out, header.original_size));
      void* d_comp = NULL;
      CUDA_CALL(hipMalloc(&d_comp, header.compressed_size));
      CUDA_CALL(hipMemcpy(d_comp, comp_buf, header.compressed_size, hipMemcpyHostToDevice));
      int ok = CUDA_DecompressWavefield(d_comp, d_out, num_floats);
      CUDA_CALL(hipFree(d_comp));
      if (!ok) {
         printf("Decompression failed for iteration %d.\n", header.iteration);
         CUDA_CALL(hipFree(d_out));
         free(comp_buf);
         break;
      }
      float* h_out = (float*)malloc(header.original_size);
      if (!h_out) { printf("Host alloc fail for decompressed output.\n"); }
      else {
         CUDA_CALL(hipMemcpy(h_out, d_out, header.original_size, hipMemcpyDeviceToHost));
         fwrite(h_out, 1, header.original_size, out_bin);
         free(h_out);
         snapshot_count++;
      }
      CUDA_CALL(hipFree(d_out));
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
   const int nx_full = sx - 2*bord - 2*absorb;
   const int ny_full = sy - 2*bord - 2*absorb;
   const int nz_full = sz - 2*bord - 2*absorb;
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
