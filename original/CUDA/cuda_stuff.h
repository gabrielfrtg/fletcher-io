#ifndef _CUDA_STUFF
#define _CUDA_STUFF

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_Initialize(const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
               float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
               float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
               float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
               float * restrict phi, float * restrict theta, 
               float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);

void CUDA_Finalize();

void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc);

void CUDA_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                    void** compressed_data, size_t* compressed_size);

// Decompress a compressed checkpoint (device buffer still lives in compression context)
// compressed_data must point to device buffer containing compressed data (currently we store only on host)
// For now, provide function to decompress last compressed buffer from host copy.
int CUDA_Decompress_to_pc(float* host_compressed, size_t compressed_size,
                          const int sx, const int sy, const int sz);

// Decompress all checkpoints from a compressed file (produces raw binary outputs per iteration)
void CUDA_DecompressCheckpointFile(const char* infile,
                                   const char* out_header,
                                   const char* out_data,
                                   int sx, int sy, int sz, int bord, int absorb,
                                   float dx, float dy, float dz, float dt_output);

#ifdef __cplusplus
}
#endif
#endif

