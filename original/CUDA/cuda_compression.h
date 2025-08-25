#ifndef __CUDA_COMPRESSION_H
#define __CUDA_COMPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize compression context
// compression_level: 0=fast, 1=default, 2=high compression
void CUDA_InitCompression(size_t max_uncompressed_size, int compression_level);

// Compress wavefield data on GPU using high-level API
// Returns compressed size, 0 on error
size_t CUDA_CompressWavefield(float* d_wavefield, size_t num_elements, void** h_compressed_output);

// Alternative: Compress using low-level API (simpler, may be more stable)
size_t CUDA_CompressWavefield_LowLevel(float* d_wavefield, size_t num_elements, void** h_compressed_output);

// Decompress previously compressed buffer into provided device wavefield buffer
int CUDA_DecompressWavefield(const void* d_compressed_buffer, float* d_output_wavefield, size_t expected_num_elements);

// Cleanup compression resources
void CUDA_FinalizeCompression();

#ifdef __cplusplus
}
#endif

#endif