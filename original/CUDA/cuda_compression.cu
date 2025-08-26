#include "cuda_defines.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_NVCOMP
#include <cuda_runtime.h>
#include <nvcomp/lz4.hpp>
#include <nvcomp/lz4.h>
#include <nvcomp.hpp>

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
    
    // Allocate compressed buffer (2x for safety)
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

extern "C" void CUDA_FinalizeCompression() {
    if (!g_comp_ctx.initialized) return;
    
    CUDA_CALL(cudaFree(g_comp_ctx.d_compressed_buffer));
    free(g_comp_ctx.h_compressed_buffer);
    CUDA_CALL(cudaStreamDestroy(g_comp_ctx.stream));
    CUDA_CALL(cudaEventDestroy(g_comp_ctx.comp_event));
    
    g_comp_ctx.initialized = 0;
}

extern "C" void CUDA_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                               void** compressed_data, size_t* compressed_size)
{
    extern float* dev_pc;
    const size_t num_elements = ((size_t)sx*sy)*sz;
    
   // Use high-level nvCOMP API wrapper
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
   CUDA_CALL(cudaMalloc(&d_comp, compressed_size));
   CUDA_CALL(cudaMemcpy(d_comp, host_compressed, compressed_size, cudaMemcpyHostToDevice));
   const size_t num_elements = ((size_t)sx*sy)*sz;
   int ok = CUDA_DecompressWavefield(d_comp, dev_pc, num_elements);
   CUDA_CALL(cudaFree(d_comp));
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
      CUDA_CALL(cudaMalloc(&d_out, header.original_size));
      void* d_comp = NULL;
      CUDA_CALL(cudaMalloc(&d_comp, header.compressed_size));
      CUDA_CALL(cudaMemcpy(d_comp, comp_buf, header.compressed_size, cudaMemcpyHostToDevice));
      int ok = CUDA_DecompressWavefield(d_comp, d_out, num_floats);
      CUDA_CALL(cudaFree(d_comp));
      if (!ok) {
         printf("Decompression failed for iteration %d.\n", header.iteration);
         CUDA_CALL(cudaFree(d_out));
         free(comp_buf);
         break;
      }
      float* h_out = (float*)malloc(header.original_size);
      if (!h_out) { printf("Host alloc fail for decompressed output.\n"); }
      else {
         CUDA_CALL(cudaMemcpy(h_out, d_out, header.original_size, cudaMemcpyDeviceToHost));
         fwrite(h_out, 1, header.original_size, out_bin);
         free(h_out);
         snapshot_count++;
      }
      CUDA_CALL(cudaFree(d_out));
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
