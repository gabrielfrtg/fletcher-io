#include "cuda_defines.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef USE_HIPCOMP
#include <hip/hip_runtime.h>
#include <hipcomp/lz4.h>

#ifndef HIPCOMP_STATUS_SUCCESS
#ifdef hipcompSuccess
#define HIPCOMP_STATUS_SUCCESS hipcompSuccess
#elif defined(nvcompSuccess)
#define HIPCOMP_STATUS_SUCCESS nvcompSuccess
#else
#define HIPCOMP_STATUS_SUCCESS 0
#endif
#endif

#if defined(hipcompStatus_t)
typedef hipcompStatus_t hipcompStatus;
#elif defined(nvcompStatus_t)
typedef nvcompStatus_t hipcompStatus;
#else
typedef int hipcompStatus;
#endif

#ifndef HIPCOMP_CALL
#define HIPCOMP_CALL(call)                                                                    \
  do {                                                                                        \
    hipcompStatus _status = call;                                                             \
    if (_status != HIPCOMP_STATUS_SUCCESS) {                                                  \
      fprintf(stderr, "hipCOMP call failed (status=%d) at %s:%d\n", (int)_status, __FILE__,  \
              __LINE__);                                                                      \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)
#endif

#ifndef HIPCOMP_BatchedLZ4DefaultOpts
#define HIPCOMP_BatchedLZ4DefaultOpts hipcompBatchedLZ4DefaultOpts
#endif

typedef struct {
  void* d_compressed_buffer;
  size_t compressed_buffer_size;

  void* h_compressed_buffer;
  size_t h_compressed_buffer_size;

  void** h_uncomp_ptrs;
  size_t* h_uncomp_sizes;
  void** h_comp_ptrs;
  size_t* h_comp_sizes;

  void** d_uncomp_ptrs;
  size_t* d_uncomp_sizes;
  void** d_comp_ptrs;
  size_t* d_comp_sizes;

  void* d_comp_temp;
  size_t comp_temp_bytes;
  void* d_decomp_temp;
  size_t decomp_temp_bytes;

  size_t* d_actual_uncomp_sizes;
  hipcompStatus* d_statuses;

  size_t* h_actual_uncomp_sizes;
  hipcompStatus* h_statuses;

  hipStream_t stream;
  hipEvent_t comp_event;

  size_t chunk_size;
  size_t max_chunks;
  size_t max_chunk_output_size;

  int compression_level;
  int initialized;
} CompressionContext;

typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t reserved;
  uint32_t chunk_size;
  uint32_t num_chunks;
  uint64_t uncompressed_bytes;
  uint64_t total_compressed_bytes;
} HipcompHeader;

static const uint32_t kHipcompMagic = 0x484C5A34u; // 'HLZ4'
static const uint16_t kHipcompVersion = 1u;

static CompressionContext g_comp_ctx = {0};

static void ensure_host_capacity(size_t required_bytes)
{
  if (g_comp_ctx.h_compressed_buffer_size >= required_bytes) return;

  void* new_buffer = realloc(g_comp_ctx.h_compressed_buffer, required_bytes);
  if (!new_buffer) {
    fprintf(stderr, "Failed to grow host compression buffer to %zu bytes\n", required_bytes);
    exit(EXIT_FAILURE);
  }
  g_comp_ctx.h_compressed_buffer = new_buffer;
  g_comp_ctx.h_compressed_buffer_size = required_bytes;
}

static void ensure_device_compressed_capacity(size_t required_bytes)
{
  if (g_comp_ctx.compressed_buffer_size >= required_bytes) return;

  if (g_comp_ctx.d_compressed_buffer) {
    CUDA_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
  }
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_compressed_buffer, required_bytes));
  g_comp_ctx.compressed_buffer_size = required_bytes;
}

static void* host_realloc_checked(void* ptr, size_t bytes, const char* label)
{
  void* new_ptr = realloc(ptr, bytes);
  if (!new_ptr) {
    fprintf(stderr, "hipCOMP: failed to allocate %s (%zu bytes)\n", label, bytes);
    exit(EXIT_FAILURE);
  }
  return new_ptr;
}

static void ensure_chunk_resources(size_t chunk_size, size_t required_chunks)
{
  if (chunk_size == 0) {
    chunk_size = 64 * 1024;
  }
  if (required_chunks == 0) {
    required_chunks = 1;
  }

  const bool chunk_changed = (g_comp_ctx.chunk_size != chunk_size);
  const bool need_more_chunks = (required_chunks > g_comp_ctx.max_chunks);

  if (!chunk_changed && !need_more_chunks && g_comp_ctx.max_chunks != 0) {
    return;
  }

  hipcompBatchedLZ4Opts_t opts = HIPCOMP_BatchedLZ4DefaultOpts;

  g_comp_ctx.chunk_size = chunk_size;

  HIPCOMP_CALL(hipcompBatchedLZ4CompressGetMaxOutputChunkSize(
      g_comp_ctx.chunk_size, opts, &g_comp_ctx.max_chunk_output_size));

  size_t new_max_chunks = required_chunks;
  if (new_max_chunks < 1) new_max_chunks = 1;

  size_t comp_temp_bytes = 0;
  HIPCOMP_CALL(hipcompBatchedLZ4CompressGetTempSize(
      new_max_chunks, g_comp_ctx.chunk_size, opts, &comp_temp_bytes));
  if (g_comp_ctx.d_comp_temp) {
    CUDA_CALL(hipFree(g_comp_ctx.d_comp_temp));
  }
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_comp_temp, comp_temp_bytes));
  g_comp_ctx.comp_temp_bytes = comp_temp_bytes;

  size_t decomp_temp_bytes = 0;
  HIPCOMP_CALL(hipcompBatchedLZ4DecompressGetTempSize(
      new_max_chunks, g_comp_ctx.chunk_size, &decomp_temp_bytes));
  if (g_comp_ctx.d_decomp_temp) {
    CUDA_CALL(hipFree(g_comp_ctx.d_decomp_temp));
  }
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_decomp_temp, decomp_temp_bytes));
  g_comp_ctx.decomp_temp_bytes = decomp_temp_bytes;

  g_comp_ctx.h_uncomp_ptrs = (void**)host_realloc_checked(
      g_comp_ctx.h_uncomp_ptrs, new_max_chunks * sizeof(void*), "uncompressed pointer table");
  g_comp_ctx.h_comp_ptrs = (void**)host_realloc_checked(
      g_comp_ctx.h_comp_ptrs, new_max_chunks * sizeof(void*), "compressed pointer table");
  g_comp_ctx.h_uncomp_sizes = (size_t*)host_realloc_checked(
      g_comp_ctx.h_uncomp_sizes, new_max_chunks * sizeof(size_t), "uncompressed size table");
  g_comp_ctx.h_comp_sizes = (size_t*)host_realloc_checked(
      g_comp_ctx.h_comp_sizes, new_max_chunks * sizeof(size_t), "compressed size table");

  if (g_comp_ctx.d_uncomp_ptrs) CUDA_CALL(hipFree(g_comp_ctx.d_uncomp_ptrs));
  if (g_comp_ctx.d_comp_ptrs) CUDA_CALL(hipFree(g_comp_ctx.d_comp_ptrs));
  if (g_comp_ctx.d_uncomp_sizes) CUDA_CALL(hipFree(g_comp_ctx.d_uncomp_sizes));
  if (g_comp_ctx.d_comp_sizes) CUDA_CALL(hipFree(g_comp_ctx.d_comp_sizes));
  if (g_comp_ctx.d_actual_uncomp_sizes) CUDA_CALL(hipFree(g_comp_ctx.d_actual_uncomp_sizes));
  if (g_comp_ctx.d_statuses) CUDA_CALL(hipFree(g_comp_ctx.d_statuses));

  CUDA_CALL(hipMalloc(&g_comp_ctx.d_uncomp_ptrs, new_max_chunks * sizeof(void*)));
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_comp_ptrs, new_max_chunks * sizeof(void*)));
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_uncomp_sizes, new_max_chunks * sizeof(size_t)));
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_comp_sizes, new_max_chunks * sizeof(size_t)));
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_actual_uncomp_sizes, new_max_chunks * sizeof(size_t)));
  CUDA_CALL(hipMalloc(&g_comp_ctx.d_statuses, new_max_chunks * sizeof(hipcompStatus)));

  g_comp_ctx.h_actual_uncomp_sizes = (size_t*)host_realloc_checked(
      g_comp_ctx.h_actual_uncomp_sizes, new_max_chunks * sizeof(size_t),
      "actual uncompressed size table");
  g_comp_ctx.h_statuses = (hipcompStatus*)host_realloc_checked(
      g_comp_ctx.h_statuses, new_max_chunks * sizeof(hipcompStatus),
      "decompression status table");

  g_comp_ctx.max_chunks = new_max_chunks;

  const size_t device_bytes = g_comp_ctx.max_chunk_output_size * g_comp_ctx.max_chunks;
  ensure_device_compressed_capacity(device_bytes);

  const size_t host_bytes = sizeof(HipcompHeader)
      + g_comp_ctx.max_chunks * (sizeof(uint64_t) + g_comp_ctx.max_chunk_output_size);
  ensure_host_capacity(host_bytes);
}

extern "C" void CUDA_InitCompression(size_t max_uncompressed_size, int compression_level)
{
  if (g_comp_ctx.initialized) return;

  CUDA_CALL(hipStreamCreate(&g_comp_ctx.stream));
  CUDA_CALL(hipEventCreateWithFlags(&g_comp_ctx.comp_event, hipEventDisableTiming));

  g_comp_ctx.compression_level = compression_level;

  size_t chunk_size = 64 * 1024;
  if (max_uncompressed_size > 0 && max_uncompressed_size < chunk_size) {
    chunk_size = max_uncompressed_size;
  }
  if (chunk_size == 0) {
    chunk_size = 64 * 1024;
  }

  size_t max_chunks = (max_uncompressed_size + chunk_size - 1) / chunk_size;
  if (max_chunks == 0) {
    max_chunks = 1;
  }

  ensure_chunk_resources(chunk_size, max_chunks);

  g_comp_ctx.initialized = 1;
  printf("hipCOMP LZ4 compression initialized (level=%d, max %.2f MB)\n",
         compression_level,
         (g_comp_ctx.compressed_buffer_size + sizeof(HipcompHeader)) / (1024.0 * 1024.0));
}

static size_t round_up_chunks(size_t bytes)
{
  return (bytes + g_comp_ctx.chunk_size - 1) / g_comp_ctx.chunk_size;
}

extern "C" size_t CUDA_CompressWavefield(
    float* d_wavefield,
    size_t num_elements,
    void** h_compressed_output)
{
  if (!g_comp_ctx.initialized) {
    fprintf(stderr, "hipCOMP compression called before initialization\n");
    return 0;
  }

  const size_t uncompressed_bytes = num_elements * sizeof(float);
  const size_t num_chunks = round_up_chunks(uncompressed_bytes);
  ensure_chunk_resources(g_comp_ctx.chunk_size, num_chunks);

  for (size_t i = 0; i < num_chunks; ++i) {
    const size_t offset = i * g_comp_ctx.chunk_size;
    size_t chunk_bytes = g_comp_ctx.chunk_size;
    if (offset + chunk_bytes > uncompressed_bytes) {
      chunk_bytes = uncompressed_bytes - offset;
    }
    g_comp_ctx.h_uncomp_ptrs[i] = reinterpret_cast<uint8_t*>(d_wavefield) + offset;
    g_comp_ctx.h_uncomp_sizes[i] = chunk_bytes;
    g_comp_ctx.h_comp_ptrs[i] = reinterpret_cast<uint8_t*>(g_comp_ctx.d_compressed_buffer)
        + i * g_comp_ctx.max_chunk_output_size;
  }

  CUDA_CALL(hipMemcpyAsync(g_comp_ctx.d_uncomp_ptrs, g_comp_ctx.h_uncomp_ptrs,
                           num_chunks * sizeof(void*), hipMemcpyHostToDevice, g_comp_ctx.stream));
  CUDA_CALL(hipMemcpyAsync(g_comp_ctx.d_uncomp_sizes, g_comp_ctx.h_uncomp_sizes,
                           num_chunks * sizeof(size_t), hipMemcpyHostToDevice, g_comp_ctx.stream));
  CUDA_CALL(hipMemcpyAsync(g_comp_ctx.d_comp_ptrs, g_comp_ctx.h_comp_ptrs,
                           num_chunks * sizeof(void*), hipMemcpyHostToDevice, g_comp_ctx.stream));

  hipcompBatchedLZ4Opts_t opts = HIPCOMP_BatchedLZ4DefaultOpts;

  HIPCOMP_CALL(hipcompBatchedLZ4CompressAsync(
      (const void* const*)g_comp_ctx.d_uncomp_ptrs,
      g_comp_ctx.d_uncomp_sizes,
      num_chunks,
      g_comp_ctx.chunk_size,
      g_comp_ctx.d_comp_temp,
      g_comp_ctx.comp_temp_bytes,
      g_comp_ctx.d_comp_ptrs,
      g_comp_ctx.d_comp_sizes,
      opts,
      g_comp_ctx.stream));

  CUDA_CALL(hipStreamSynchronize(g_comp_ctx.stream));

  CUDA_CALL(hipMemcpy(g_comp_ctx.h_comp_sizes, g_comp_ctx.d_comp_sizes,
                      num_chunks * sizeof(size_t), hipMemcpyDeviceToHost));

  size_t payload_bytes = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    payload_bytes += g_comp_ctx.h_comp_sizes[i];
  }

  const size_t header_bytes = sizeof(HipcompHeader) + num_chunks * sizeof(uint64_t);
  const size_t total_bytes = header_bytes + payload_bytes;
  ensure_host_capacity(total_bytes);

  HipcompHeader* header = reinterpret_cast<HipcompHeader*>(g_comp_ctx.h_compressed_buffer);
  header->magic = kHipcompMagic;
  header->version = kHipcompVersion;
  header->reserved = 0;
  header->chunk_size = (uint32_t)g_comp_ctx.chunk_size;
  header->num_chunks = (uint32_t)num_chunks;
  header->uncompressed_bytes = uncompressed_bytes;
  header->total_compressed_bytes = payload_bytes;

  uint64_t* chunk_sizes = reinterpret_cast<uint64_t*>(header + 1);
  for (size_t i = 0; i < num_chunks; ++i) {
    chunk_sizes[i] = g_comp_ctx.h_comp_sizes[i];
  }

  uint8_t* dst = reinterpret_cast<uint8_t*>(chunk_sizes + num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    const uint8_t* src = reinterpret_cast<uint8_t*>(g_comp_ctx.h_comp_ptrs[i]);
    CUDA_CALL(hipMemcpy(dst, src, g_comp_ctx.h_comp_sizes[i], hipMemcpyDeviceToHost));
    dst += g_comp_ctx.h_comp_sizes[i];
  }

  CUDA_CALL(hipEventRecord(g_comp_ctx.comp_event, g_comp_ctx.stream));
  CUDA_CALL(hipEventSynchronize(g_comp_ctx.comp_event));

  *h_compressed_output = g_comp_ctx.h_compressed_buffer;
  return total_bytes;
}

static int parse_header(const void* device_buffer, HipcompHeader* host_header, uint64_t** chunk_sizes_out)
{
  CUDA_CALL(hipMemcpy(host_header, device_buffer, sizeof(HipcompHeader), hipMemcpyDeviceToHost));
  if (host_header->magic != kHipcompMagic) {
    fprintf(stderr, "hipCOMP: invalid magic in compressed stream (0x%x)\n", host_header->magic);
    return 0;
  }
  if (host_header->version != kHipcompVersion) {
    fprintf(stderr, "hipCOMP: unsupported version %u\n", host_header->version);
    return 0;
  }
  const size_t table_bytes = host_header->num_chunks * sizeof(uint64_t);
  *chunk_sizes_out = (uint64_t*)malloc(table_bytes);
  if (!*chunk_sizes_out) {
    fprintf(stderr, "hipCOMP: failed to allocate chunk size table\n");
    return 0;
  }
  CUDA_CALL(hipMemcpy(*chunk_sizes_out,
                      reinterpret_cast<const uint8_t*>(device_buffer) + sizeof(HipcompHeader),
                      table_bytes,
                      hipMemcpyDeviceToHost));
  return 1;
}

extern "C" int CUDA_DecompressWavefield(
    const void* d_compressed_buffer,
    float* d_output_wavefield,
    size_t expected_num_elements)
{
  if (!g_comp_ctx.initialized) {
    const size_t required_bytes = expected_num_elements * sizeof(float);
    CUDA_InitCompression(required_bytes, 0);
  }

  HipcompHeader header;
  uint64_t* chunk_sizes = NULL;
  if (!parse_header(d_compressed_buffer, &header, &chunk_sizes)) {
    free(chunk_sizes);
    return 0;
  }

  if (header.uncompressed_bytes != expected_num_elements * sizeof(float)) {
    fprintf(stderr,
            "hipCOMP: decompressed size (%llu) does not match expected (%zu)\n",
            (unsigned long long)header.uncompressed_bytes,
            expected_num_elements * sizeof(float));
  }

  ensure_chunk_resources(header.chunk_size, header.num_chunks);

  const uint8_t* base_ptr = reinterpret_cast<const uint8_t*>(d_compressed_buffer);
  const uint8_t* payload_ptr = base_ptr + sizeof(HipcompHeader) + header.num_chunks * sizeof(uint64_t);

  for (size_t i = 0; i < header.num_chunks; ++i) {
    g_comp_ctx.h_comp_ptrs[i] = const_cast<uint8_t*>(payload_ptr);
    g_comp_ctx.h_comp_sizes[i] = (size_t)chunk_sizes[i];
    payload_ptr += chunk_sizes[i];

    size_t chunk_bytes = g_comp_ctx.chunk_size;
    const size_t offset = i * g_comp_ctx.chunk_size;
    if (offset + chunk_bytes > header.uncompressed_bytes) {
      chunk_bytes = header.uncompressed_bytes - offset;
    }
    g_comp_ctx.h_uncomp_ptrs[i] = reinterpret_cast<uint8_t*>(d_output_wavefield) + offset;
    g_comp_ctx.h_uncomp_sizes[i] = chunk_bytes;
  }

  const size_t consumed = static_cast<size_t>(payload_ptr -
      (base_ptr + sizeof(HipcompHeader) + header.num_chunks * sizeof(uint64_t)));
  if (header.total_compressed_bytes != consumed) {
    fprintf(stderr,
            "hipCOMP: compressed payload size mismatch (header=%llu, consumed=%zu)\n",
            (unsigned long long)header.total_compressed_bytes,
            consumed);
  }

  CUDA_CALL(hipMemcpyAsync(g_comp_ctx.d_comp_ptrs, g_comp_ctx.h_comp_ptrs,
                           header.num_chunks * sizeof(void*), hipMemcpyHostToDevice, g_comp_ctx.stream));
  CUDA_CALL(hipMemcpyAsync(g_comp_ctx.d_comp_sizes, g_comp_ctx.h_comp_sizes,
                           header.num_chunks * sizeof(size_t), hipMemcpyHostToDevice, g_comp_ctx.stream));
  CUDA_CALL(hipMemcpyAsync(g_comp_ctx.d_uncomp_ptrs, g_comp_ctx.h_uncomp_ptrs,
                           header.num_chunks * sizeof(void*), hipMemcpyHostToDevice, g_comp_ctx.stream));
  CUDA_CALL(hipMemcpyAsync(g_comp_ctx.d_uncomp_sizes, g_comp_ctx.h_uncomp_sizes,
                           header.num_chunks * sizeof(size_t), hipMemcpyHostToDevice, g_comp_ctx.stream));

  CUDA_CALL(hipMemset(g_comp_ctx.d_statuses, 0,
                      header.num_chunks * sizeof(hipcompStatus)));

  HIPCOMP_CALL(hipcompBatchedLZ4DecompressAsync(
      (const void* const*)g_comp_ctx.d_comp_ptrs,
      g_comp_ctx.d_comp_sizes,
      g_comp_ctx.d_uncomp_sizes,
      g_comp_ctx.d_actual_uncomp_sizes,
      header.num_chunks,
      g_comp_ctx.d_decomp_temp,
      g_comp_ctx.decomp_temp_bytes,
      (void* const*)g_comp_ctx.d_uncomp_ptrs,
      g_comp_ctx.d_statuses,
      g_comp_ctx.stream));

  CUDA_CALL(hipStreamSynchronize(g_comp_ctx.stream));

  CUDA_CALL(hipMemcpy(g_comp_ctx.h_actual_uncomp_sizes,
                      g_comp_ctx.d_actual_uncomp_sizes,
                      header.num_chunks * sizeof(size_t),
                      hipMemcpyDeviceToHost));
  CUDA_CALL(hipMemcpy(g_comp_ctx.h_statuses,
                      g_comp_ctx.d_statuses,
                      header.num_chunks * sizeof(hipcompStatus),
                      hipMemcpyDeviceToHost));

  int success = 1;
  for (size_t i = 0; i < header.num_chunks; ++i) {
    if (g_comp_ctx.h_statuses[i] != HIPCOMP_STATUS_SUCCESS) {
      fprintf(stderr, "hipCOMP: decompression failed for chunk %zu (status=%d)\n",
              i, (int)g_comp_ctx.h_statuses[i]);
      success = 0;
      break;
    }
    if (g_comp_ctx.h_actual_uncomp_sizes[i] != g_comp_ctx.h_uncomp_sizes[i]) {
      fprintf(stderr,
              "hipCOMP: chunk %zu decompressed size mismatch (expected=%zu, actual=%zu)\n",
              i,
              g_comp_ctx.h_uncomp_sizes[i],
              g_comp_ctx.h_actual_uncomp_sizes[i]);
      success = 0;
      break;
    }
  }

  free(chunk_sizes);
  return success;
}

extern "C" void CUDA_FinalizeCompression()
{
  if (!g_comp_ctx.initialized) return;

  if (g_comp_ctx.d_compressed_buffer) CUDA_CALL(hipFree(g_comp_ctx.d_compressed_buffer));
  if (g_comp_ctx.d_comp_temp) CUDA_CALL(hipFree(g_comp_ctx.d_comp_temp));
  if (g_comp_ctx.d_decomp_temp) CUDA_CALL(hipFree(g_comp_ctx.d_decomp_temp));
  if (g_comp_ctx.d_uncomp_ptrs) CUDA_CALL(hipFree(g_comp_ctx.d_uncomp_ptrs));
  if (g_comp_ctx.d_uncomp_sizes) CUDA_CALL(hipFree(g_comp_ctx.d_uncomp_sizes));
  if (g_comp_ctx.d_comp_ptrs) CUDA_CALL(hipFree(g_comp_ctx.d_comp_ptrs));
  if (g_comp_ctx.d_comp_sizes) CUDA_CALL(hipFree(g_comp_ctx.d_comp_sizes));
  if (g_comp_ctx.d_actual_uncomp_sizes) CUDA_CALL(hipFree(g_comp_ctx.d_actual_uncomp_sizes));
  if (g_comp_ctx.d_statuses) CUDA_CALL(hipFree(g_comp_ctx.d_statuses));

  free(g_comp_ctx.h_compressed_buffer);
  free(g_comp_ctx.h_uncomp_ptrs);
  free(g_comp_ctx.h_uncomp_sizes);
  free(g_comp_ctx.h_comp_ptrs);
  free(g_comp_ctx.h_comp_sizes);
  free(g_comp_ctx.h_actual_uncomp_sizes);
  free(g_comp_ctx.h_statuses);

  CUDA_CALL(hipEventDestroy(g_comp_ctx.comp_event));
  CUDA_CALL(hipStreamDestroy(g_comp_ctx.stream));

  memset(&g_comp_ctx, 0, sizeof(g_comp_ctx));
}

extern "C" void CUDA_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                               void** compressed_data, size_t* compressed_size)
{
  extern float* dev_pc;
  const size_t num_elements = ((size_t)sx * sy) * sz;

  *compressed_size = CUDA_CompressWavefield(dev_pc, num_elements, compressed_data);

  if (*compressed_size > 0) {
    const double original_mb = (num_elements * sizeof(float)) / (1024.0 * 1024.0);
    const double compressed_mb = (*compressed_size) / (1024.0 * 1024.0);
    const double ratio = original_mb / compressed_mb;
    printf("Compressed checkpoint: %.2f MB -> %.2f MB (ratio: %.2fx)\n",
           original_mb, compressed_mb, ratio);
  } else {
    printf("Warning: Compression disabled or failed; using uncompressed data\n");
  }
}

extern "C" int CUDA_Decompress_to_pc(float* host_compressed, size_t compressed_size,
                                      const int sx, const int sy, const int sz)
{
  if (compressed_size == 0) return 0;

  extern float* dev_pc;

  void* d_comp = NULL;
  CUDA_CALL(hipMalloc(&d_comp, compressed_size));
  CUDA_CALL(hipMemcpy(d_comp, host_compressed, compressed_size, hipMemcpyHostToDevice));

  const size_t num_elements = ((size_t)sx * sy) * sz;
  const int ok = CUDA_DecompressWavefield(d_comp, dev_pc, num_elements);

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
    if (r != 1) break;

    if (snapshot_count == 0) first_time = header.timestamp;
    else if (snapshot_count == 1) second_time = header.timestamp;

    if (header.compressed_size == 0 || header.compressed_size > (1ULL << 40)) {
      printf("Invalid compressed_size in header, aborting decompression loop.\n");
      break;
    }

    void* comp_buf = malloc(header.compressed_size);
    if (!comp_buf) {
      printf("Alloc fail for compressed buffer.\n");
      break;
    }

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
    if (!h_out) {
      printf("Host alloc fail for decompressed output.\n");
    } else {
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

  FILE* out_hdr = fopen(out_header, "w");
  if (!out_hdr) {
    printf("Could not open header file %s for writing.\n", out_header);
    return;
  }

  const int nx_full = sx - 2 * bord - 2 * absorb;
  const int ny_full = sy - 2 * bord - 2 * absorb;
  const int nz_full = sz - 2 * bord - 2 * absorb;
  float inferred_dt = dt_output;
  if (snapshot_count > 1 && second_time > first_time) {
    inferred_dt = second_time - first_time;
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

  printf("File-level decompression complete: %d snapshots -> %s (%s).\n",
         snapshot_count, out_data, out_header);
}

#endif // USE_HIPCOMP
