#include "hip_defines.h"
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

// For Unified Memory version, d_wavefield is already accessible from device
extern "C" size_t HIP_CompressWavefield(
    float* d_wavefield,          // Unified Memory pointer to wavefield
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

// Decompress a compressed buffer back into a provided device buffer
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

extern "C" void HIP_FinalizeCompression() {
    if (!g_comp_ctx.initialized) return;

    HIP_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
    free(g_comp_ctx.h_compressed_buffer);
    HIP_CALL(hipStreamDestroy(g_comp_ctx.stream));
    HIP_CALL(hipEventDestroy(g_comp_ctx.comp_event));

    g_comp_ctx.initialized = 0;
}

extern "C" void HIP_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                               float* pc, void** compressed_data, size_t* compressed_size)
{
    const size_t num_elements = ((size_t)sx*sy)*sz;

   // Use hipCOMP batched LZ4 API wrapper
   *compressed_size = 0;
   *compressed_size = HIP_CompressWavefield(pc, num_elements, compressed_data);

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
                    const int sx, const int sy, const int sz, float* pc)
{
   if (compressed_size == 0) return 0;
   // Upload compressed data to device temp buffer
   void* d_comp = NULL;
   HIP_CALL(hipMalloc(&d_comp, compressed_size));
   HIP_CALL(hipMemcpy(d_comp, host_compressed, compressed_size, hipMemcpyHostToDevice));
   const size_t num_elements = ((size_t)sx*sy)*sz;
   int ok = HIP_DecompressWavefield(d_comp, pc, num_elements);
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
