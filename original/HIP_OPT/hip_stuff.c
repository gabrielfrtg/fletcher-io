#include "hip_defines.h"
#include "hip_stuff.h"
#ifdef USE_HIPCOMP
#include "hip_compression.h"
#endif

static size_t sxsy=0;

void HIP_Initialize(const int sx, const int sy, const int sz, const int bord,
	       float dx, float dy, float dz, float dt,
	       float *  ch1dxx, float *  ch1dyy, float *  ch1dzz, 
	       float *  ch1dxy, float *  ch1dyz, float *  ch1dxz, 
	       float *  v2px, float *  v2pz, float *  v2sz, float *  v2pn,
	       float *  vpz, float *  vsv, float *  epsilon, float *  delta,
	       float *  phi, float *  theta, 
	       float *  pp, float *  pc, float *  qp, float *  qc)
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
  HIP_CALL(hipGetDeviceCount(&deviceCount));
  const int device=0;
  hipDeviceProp_t deviceProp;
  HIP_CALL(hipGetDeviceProperties(&deviceProp, device));
  printf("HIP source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  HIP_CALL(hipSetDevice(device));


  // Check sx,sy values - removed constraints for better flexibility
 /* if (sx%BSIZE_X != 0)
  {
     printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
     exit(1);
  } 
  if (sy%BSIZE_Y != 0)
  {
     printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
     exit(1);
  } 
*/
   sxsy=sx*sy; // one plan
   const size_t sxsysz=sxsy*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   const size_t msize_vol_extra=msize_vol+2*sxsy*sizeof(float); // 2 extra plans for wave fields
   HIP_CALL(hipMalloc(&dev_vpz, msize_vol));
   HIP_CALL(hipMemcpy(dev_vpz, vpz, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_vsv, msize_vol));
   HIP_CALL(hipMemcpy(dev_vsv, vsv, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_epsilon, msize_vol));
   HIP_CALL(hipMemcpy(dev_epsilon, epsilon, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_delta, msize_vol));
   HIP_CALL(hipMemcpy(dev_delta, delta, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_phi, msize_vol));
   HIP_CALL(hipMemcpy(dev_phi, phi, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_theta, msize_vol));
   HIP_CALL(hipMemcpy(dev_theta, theta, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_ch1dxx, msize_vol));
   HIP_CALL(hipMemcpy(dev_ch1dxx, ch1dxx, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_ch1dyy, msize_vol));
   HIP_CALL(hipMemcpy(dev_ch1dyy, ch1dyy, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_ch1dzz, msize_vol));
   HIP_CALL(hipMemcpy(dev_ch1dzz, ch1dzz, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_ch1dxy, msize_vol));
   HIP_CALL(hipMemcpy(dev_ch1dxy, ch1dxy, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_ch1dyz, msize_vol));
   HIP_CALL(hipMemcpy(dev_ch1dyz, ch1dyz, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_ch1dxz, msize_vol));
   HIP_CALL(hipMemcpy(dev_ch1dxz, ch1dxz, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_v2px, msize_vol));
   HIP_CALL(hipMemcpy(dev_v2px, v2px, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_v2pz, msize_vol));
   HIP_CALL(hipMemcpy(dev_v2pz, v2pz, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_v2sz, msize_vol));
   HIP_CALL(hipMemcpy(dev_v2sz, v2sz, msize_vol, hipMemcpyHostToDevice));
   HIP_CALL(hipMalloc(&dev_v2pn, msize_vol));
   HIP_CALL(hipMemcpy(dev_v2pn, v2pn, msize_vol, hipMemcpyHostToDevice));

   // Wave field arrays with an extra plan
   HIP_CALL(hipMalloc(&dev_pp, msize_vol_extra));
   HIP_CALL(hipMemset(dev_pp, 0, msize_vol_extra));
   HIP_CALL(hipMalloc(&dev_pc, msize_vol_extra));
   HIP_CALL(hipMemset(dev_pc, 0, msize_vol_extra));
   HIP_CALL(hipMalloc(&dev_qp, msize_vol_extra));
   HIP_CALL(hipMemset(dev_qp, 0, msize_vol_extra));
   HIP_CALL(hipMalloc(&dev_qc, msize_vol_extra));
   HIP_CALL(hipMemset(dev_qc, 0, msize_vol_extra));
   dev_pp+=sxsy;
   dev_pc+=sxsy;
   dev_qp+=sxsy;
   dev_qc+=sxsy;

   HIP_CALL(hipMalloc(&dev_pDx, msize_vol));
   HIP_CALL(hipMemset(dev_pDx, 0, msize_vol));
   HIP_CALL(hipMalloc(&dev_pDy, msize_vol));
   HIP_CALL(hipMemset(dev_pDy, 0, msize_vol));
   HIP_CALL(hipMalloc(&dev_qDx, msize_vol));
   HIP_CALL(hipMemset(dev_qDx, 0, msize_vol));
   HIP_CALL(hipMalloc(&dev_qDy, msize_vol));
   HIP_CALL(hipMemset(dev_qDy, 0, msize_vol));



  HIP_CALL(hipGetLastError());
  HIP_CALL(hipDeviceSynchronize());
  printf("GPU memory usage = %ld MiB\n", 21*msize_vol/1024/1024);

#ifdef USE_HIPCOMP
   const size_t max_uncompressed = ((size_t)sx*sy)*sz * sizeof(float);
   HIP_InitCompression(max_uncompressed, 0); // 0=fast compression
#endif

}


void HIP_Finalize()
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

   HIP_CALL(hipFree(dev_vpz));
   HIP_CALL(hipFree(dev_vsv));
   HIP_CALL(hipFree(dev_epsilon));
   HIP_CALL(hipFree(dev_delta));
   HIP_CALL(hipFree(dev_phi));
   HIP_CALL(hipFree(dev_theta));
   HIP_CALL(hipFree(dev_ch1dxx));
   HIP_CALL(hipFree(dev_ch1dyy));
   HIP_CALL(hipFree(dev_ch1dzz));
   HIP_CALL(hipFree(dev_ch1dxy));
   HIP_CALL(hipFree(dev_ch1dyz));
   HIP_CALL(hipFree(dev_ch1dxz));
   HIP_CALL(hipFree(dev_v2px));
   HIP_CALL(hipFree(dev_v2pz));
   HIP_CALL(hipFree(dev_v2sz));
   HIP_CALL(hipFree(dev_v2pn));
   HIP_CALL(hipFree(dev_pp));
   HIP_CALL(hipFree(dev_pc));
   HIP_CALL(hipFree(dev_qp));
   HIP_CALL(hipFree(dev_qc));
   HIP_CALL(hipFree(dev_pDx));
   HIP_CALL(hipFree(dev_qDx));
   HIP_CALL(hipFree(dev_pDy));
   HIP_CALL(hipFree(dev_qDy));

#ifdef USE_HIPCOMP
   HIP_FinalizeCompression();
#endif

   printf("HIP_Finalize: SUCCESS\n");
}



void HIP_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   extern float* dev_pc;
   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   if (pc) HIP_CALL(hipMemcpy(pc, dev_pc, msize_vol, hipMemcpyDeviceToHost));
}
