#include "cuda_defines.h"
#include "cuda_stuff.h"

static size_t sxsy=0;

void CUDA_Initialize(const int sx, const int sy, const int sz, const int bord,
	       float dx, float dy, float dz, float dt,
	       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
	       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
	       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	       float * restrict phi, float * restrict theta, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

   extern float* dev_pDx;
   extern float* dev_pDy;
   extern float* dev_qDx;
   extern float* dev_qDy;
   extern float* dev_vpz;
   extern float* dev_vsv;
   extern float* dev_epsilon;
   extern float* dev_delta;
   extern float* dev_phi;
   extern float* dev_theta;
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

 
  int deviceCount;
  CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  const int device=0;
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  CUDA_CALL(cudaSetDevice(device));


  // Check sx,sy values
  if (sx%BSIZE_X != 0)
  {
     printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
     exit(1);
  } 
  if (sy%BSIZE_Y != 0)
  {
     printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
     exit(1);
  } 

   sxsy=sx*sy; // one plan
   const size_t sxsysz=sxsy*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   const size_t msize_vol_extra=msize_vol+2*sxsy*sizeof(float); // 2 extra plans for wave fields
   CUDA_CALL(cudaMalloc(&dev_vpz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_vpz, vpz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_vsv, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_vsv, vsv, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_epsilon, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_epsilon, epsilon, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_delta, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_delta, delta, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_phi, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_phi, phi, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_theta, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_theta, theta, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dxx, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dxx, ch1dxx, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dyy, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dyy, ch1dyy, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dzz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dzz, ch1dzz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dxy, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dxy, ch1dxy, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dyz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dyz, ch1dyz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_ch1dxz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_ch1dxz, ch1dxz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2px, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2px, v2px, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2pz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2pz, v2pz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2sz, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2sz, v2sz, msize_vol, cudaMemcpyHostToDevice));
   CUDA_CALL(cudaMalloc(&dev_v2pn, msize_vol));
   CUDA_CALL(cudaMemcpy(dev_v2pn, v2pn, msize_vol, cudaMemcpyHostToDevice));

   // Wave field arrays with an extra plan
   CUDA_CALL(cudaMalloc(&dev_pp, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_pp, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_pc, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_pc, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_qp, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_qp, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_qc, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_qc, 0, msize_vol_extra));
   dev_pp+=sxsy;
   dev_pc+=sxsy;
   dev_qp+=sxsy;
   dev_qc+=sxsy;

   CUDA_CALL(cudaMalloc(&dev_pDx, msize_vol));
   CUDA_CALL(cudaMemset(dev_pDx, 0, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_pDy, msize_vol));
   CUDA_CALL(cudaMemset(dev_pDy, 0, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_qDx, msize_vol));
   CUDA_CALL(cudaMemset(dev_qDx, 0, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_qDy, msize_vol));
   CUDA_CALL(cudaMemset(dev_qDy, 0, msize_vol));



  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  printf("GPU memory usage = %ld MiB\n", 21*msize_vol/1024/1024);

}


void CUDA_Finalize()
{

   extern float* dev_vpz;
   extern float* dev_vsv;
   extern float* dev_epsilon;
   extern float* dev_delta;
   extern float* dev_phi;
   extern float* dev_theta;
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

   dev_pp-=sxsy;
   dev_pc-=sxsy;
   dev_qp-=sxsy;
   dev_qc-=sxsy;

   CUDA_CALL(cudaFree(dev_vpz));
   CUDA_CALL(cudaFree(dev_vsv));
   CUDA_CALL(cudaFree(dev_epsilon));
   CUDA_CALL(cudaFree(dev_delta));
   CUDA_CALL(cudaFree(dev_phi));
   CUDA_CALL(cudaFree(dev_theta));
   CUDA_CALL(cudaFree(dev_ch1dxx));
   CUDA_CALL(cudaFree(dev_ch1dyy));
   CUDA_CALL(cudaFree(dev_ch1dzz));
   CUDA_CALL(cudaFree(dev_ch1dxy));
   CUDA_CALL(cudaFree(dev_ch1dyz));
   CUDA_CALL(cudaFree(dev_ch1dxz));
   CUDA_CALL(cudaFree(dev_v2px));
   CUDA_CALL(cudaFree(dev_v2pz));
   CUDA_CALL(cudaFree(dev_v2sz));
   CUDA_CALL(cudaFree(dev_v2pn));
   CUDA_CALL(cudaFree(dev_pp));
   CUDA_CALL(cudaFree(dev_pc));
   CUDA_CALL(cudaFree(dev_qp));
   CUDA_CALL(cudaFree(dev_qc));
   CUDA_CALL(cudaFree(dev_pDx));
   CUDA_CALL(cudaFree(dev_qDx));
   CUDA_CALL(cudaFree(dev_pDy));
   CUDA_CALL(cudaFree(dev_qDy));

   printf("CUDA_Finalize: SUCCESS\n");
}



void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   extern float* dev_pc;
   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   if (pc) CUDA_CALL(cudaMemcpy(pc, dev_pc, msize_vol, cudaMemcpyDeviceToHost));
}
