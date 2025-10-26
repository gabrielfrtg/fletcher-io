#ifndef __HIP_CUDA_COMPRESSION_H
#define __HIP_CUDA_COMPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_InitCompression(size_t max_uncompressed_size, int compression_level);

void CUDA_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                    void** compressed_data, size_t* compressed_size);

int CUDA_Decompress_to_pc(float* host_compressed, size_t compressed_size,
                          const int sx, const int sy, const int sz);

void CUDA_DecompressCheckpointFile(const char* infile,
                                   const char* out_header,
                                   const char* out_data,
                                   int sx, int sy, int sz, int bord, int absorb,
                                   float dx, float dy, float dz, float dt_output);

void CUDA_FinalizeCompression();

#ifdef __cplusplus
}
#endif

#endif
