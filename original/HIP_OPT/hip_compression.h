#ifndef __HIP_COMPRESSION_H
#define __HIP_COMPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize compression context
// compression_level: 0=fast, 1=default, 2=high compression
void HIP_InitCompression(size_t max_uncompressed_size, int compression_level);

#ifdef COMP_ALGO_ZFP
// ZFP-specific functions

// Set ZFP dimensions (must be called before compression for 3D arrays)
void HIP_SetZfpDimensions(size_t nx, size_t ny, size_t nz);

// Enable/disable error validation (for benchmarking lossy compression quality)
// When enabled, saves original data before compression and computes error after decompression
void HIP_SetZfpValidation(int enable);

// Get error statistics after decompression (only valid if validation was enabled)
void HIP_GetZfpErrorStats(double* max_abs, double* max_rel, double* rmse, double* psnr);

// Print error statistics to stdout
void HIP_PrintZfpErrorStats();
#endif

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
