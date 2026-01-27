#ifndef __HIP_COMPRESSION_H
#define __HIP_COMPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize compression context
// compression_level: 0=fast, 1=default, 2=high compression
void HIP_InitCompression(size_t max_uncompressed_size, int compression_level);

void HIP_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                    void** compressed_data, size_t* compressed_size);

// Decompress a compressed checkpoint (device buffer still lives in compression context)
// compressed_data must point to device buffer containing compressed data (currently we store only on host)
// For now, provide function to decompress last compressed buffer from host copy.
int HIP_Decompress_to_pc(float* host_compressed, size_t compressed_size,
                          const int sx, const int sy, const int sz);

// Decompress all checkpoints from a compressed file (produces raw binary outputs per iteration)
void HIP_DecompressCheckpointFile(const char* infile,
                                   const char* out_header,
                                   const char* out_data,
                                   int sx, int sy, int sz, int bord, int absorb,
                                   float dx, float dy, float dz, float dt_output);

// Cleanup compression resources
void HIP_FinalizeCompression();

#ifdef __cplusplus
}
#endif

#endif
