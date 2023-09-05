#include "cuda_defines.h"
#include "cuda_propagate.h"
#include "../derivatives.h"
#include "../map.h"

__global__ void kernel_Propagate(const int sx, const int sy, const int sz, const int bord,
				 const float dx, const float dy, const float dz, const float dt,  
				 const int it, const float * const restrict ch1dxx, 
				 const float * const restrict ch1dyy, float * restrict ch1dzz, 
				 float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
				 float * restrict v2px, float * restrict v2pz, float * restrict v2sz, 
				 float * restrict v2pn, float * restrict pp, float * restrict pc, 
				 float * restrict qp, float * restrict qc)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  const int iy=blockIdx.y * blockDim.y + threadIdx.y;

#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP

    // solve both equations in all internal grid points, 
    // including absortion zone
    
    for (int iz=bord+1; iz<sz-bord-1; iz++) {

#define SAMPLE_LOOP
#include "../sample.h"
#undef SAMPLE_LOOP

    }
}


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void CUDA_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it, 
		    float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
  
  extern float* dev_ch1dxx;
  extern float* dev_ch1dyy;
  extern float* dev_ch1dzz;
  extern float* dev_ch1dxy;
  extern float* dev_ch1dyz;
  extern float* dev_ch1dxz;
  extern float* dev_v2px;
  extern float* dev_v2pz;
  extern float* dev_v2sz;
  extern float* dev_v2pn;
  extern float* dev_pp;
  extern float* dev_pc;
  extern float* dev_qp;
  extern float* dev_qc;
  extern float* dev_pDx;
  extern float* dev_pDy;
  extern float* dev_qDx;
  extern float* dev_qDy;
  
  
  dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
  dim3 numBlocks(sx/threadsPerBlock.x, sy/threadsPerBlock.y);
  
  kernel_Propagate<<<numBlocks, threadsPerBlock>>> (  sx,   sy,   sz,   bord,
						      dx,   dy,   dz,   dt,   it, 
						      dev_ch1dxx,  dev_ch1dyy,  dev_ch1dzz, 
						      dev_ch1dxy,  dev_ch1dyz,  dev_ch1dxz, 
						      dev_v2px,  dev_v2pz,  dev_v2sz,  dev_v2pn,
						      dev_pp,  dev_pc,  dev_qp,  dev_qc);
  CUDA_CALL(cudaGetLastError());
  CUDA_SwapArrays(&dev_pp, &dev_pc, &dev_qp, &dev_qc);
  CUDA_CALL(cudaDeviceSynchronize());
}

// swap array pointers on time forward array propagation
void CUDA_SwapArrays(float **pp, float **pc, float **qp, float **qc) {
  float *tmp;
  
  tmp=*pp;
  *pp=*pc;
  *pc=tmp;
  
  tmp=*qp;
  *qp=*qc;
  *qc=tmp;
}
