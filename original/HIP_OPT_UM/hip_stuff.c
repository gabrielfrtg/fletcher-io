#include "hip_defines.h"
#include "hip_stuff.h"
#ifdef USE_HIPCOMP
#include "hip_compression.h"
#endif
#include <string.h>

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

 
  int deviceCount;
  HIP_CALL(hipGetDeviceCount(&deviceCount));
  const int device=0;
  hipDeviceProp_t deviceProp;
  HIP_CALL(hipGetDeviceProperties(&deviceProp, device));
  printf("HIP source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  HIP_CALL(hipSetDevice(device));


   sxsy=sx*sy; // one plan
   const size_t sxsysz=sxsy*sz;
   const size_t msize_vol=sxsysz*sizeof(float);
   const size_t msize_vol_extra=msize_vol+2*sxsy*sizeof(float); // 2 extra plans for wave fields
   
   pp+=sxsy;
   pc+=sxsy;
   qp+=sxsy;
   qc+=sxsy;

  HIP_CALL(hipDeviceSynchronize());
  printf("GPU memory usage = %ld MiB\n", 21*msize_vol/1024/1024);

#ifdef USE_HIPCOMP
   const size_t max_uncompressed = ((size_t)sx*sy)*sz * sizeof(float);
   HIP_InitCompression(max_uncompressed, 0); // 0=fast compression
#endif

}


void HIP_Finalize(const int sx, const int sy, const int sz, const int bord, float dx, float dy, float dz, float dt, float *ch1dxx, float *ch1dyy, float *ch1dzz, float *ch1dxy, float *ch1dyz, float *ch1dxz, float *v2px, float *v2pz, float *v2sz, float *v2pn, float *vpz, float *vsv, float *epsilon, float *delta, float *phi, float *theta, float *pp, float *pc, float *qp, float *qc)
{

   pp-=sxsy;
   pc-=sxsy;
   qp-=sxsy;
   qc-=sxsy;

   HIP_CALL(hipFree(vpz));
   HIP_CALL(hipFree(vsv));
   HIP_CALL(hipFree(epsilon));
   HIP_CALL(hipFree(delta));
   HIP_CALL(hipFree(phi));
   HIP_CALL(hipFree(theta));
   HIP_CALL(hipFree(ch1dxx));
   HIP_CALL(hipFree(ch1dyy));
   HIP_CALL(hipFree(ch1dzz));
   HIP_CALL(hipFree(ch1dxy));
   HIP_CALL(hipFree(ch1dyz));
   HIP_CALL(hipFree(ch1dxz));
   HIP_CALL(hipFree(v2px));
   HIP_CALL(hipFree(v2pz));
   HIP_CALL(hipFree(v2sz));
   HIP_CALL(hipFree(v2pn));
   HIP_CALL(hipFree(pp));
   HIP_CALL(hipFree(pc));
   HIP_CALL(hipFree(qp));
   HIP_CALL(hipFree(qc));

#ifdef USE_HIPCOMP
   HIP_FinalizeCompression();
#endif

   printf("HIP_Finalize: SUCCESS\n");
}



void HIP_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   // With unified memory, no explicit copy is needed
   // HIP_CALL(hipDeviceSynchronize());
}


void HIP_Allocate_Model_Variables(float **  ch1dxx, float **  ch1dyy, float **  ch1dzz, float **  ch1dxy,
   float **  ch1dyz, float **  ch1dxz, float **  v2px, float **  v2pz, float **  v2sz,
   float **  v2pn, int sx, int sy, int sz){
   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   HIP_CALL(hipMallocManaged(ch1dxx, msize_vol));
   HIP_CALL(hipMallocManaged(ch1dyy, msize_vol));
   HIP_CALL(hipMallocManaged(ch1dzz, msize_vol));
   HIP_CALL(hipMallocManaged(ch1dxy, msize_vol));
   HIP_CALL(hipMallocManaged(ch1dyz, msize_vol));
   HIP_CALL(hipMallocManaged(ch1dxz, msize_vol));
   HIP_CALL(hipMallocManaged(v2px, msize_vol));
   HIP_CALL(hipMallocManaged(v2pz, msize_vol));
   HIP_CALL(hipMallocManaged(v2sz, msize_vol));
   HIP_CALL(hipMallocManaged(v2pn, msize_vol));
}


void HIP_Allocate_main(float **  vpz, float **  vsv, float **  epsilon, float **  delta, 
   float **  phi, float **  theta, float **  pp, float **  pc, float **  qp, 
   float **  qc, int sx, int sy, int sz){
   int sxsy = sx*sy;
   const size_t sxsysz=((size_t)sx*sy)*sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   const size_t msize_vol_extra = msize_vol+2*sx*sy*sizeof(float); // 2 extra plans for wave fields
   HIP_CALL(hipMallocManaged(vpz, msize_vol));
   HIP_CALL(hipMallocManaged(vsv, msize_vol));
   HIP_CALL(hipMallocManaged(epsilon, msize_vol));
   HIP_CALL(hipMallocManaged(delta, msize_vol));
   HIP_CALL(hipMallocManaged(phi, msize_vol));
   HIP_CALL(hipMallocManaged(theta, msize_vol));

   HIP_CALL(hipMallocManaged(pp, msize_vol_extra));
   HIP_CALL(hipMallocManaged(pc, msize_vol_extra));
   HIP_CALL(hipMallocManaged(qp, msize_vol_extra));
   HIP_CALL(hipMallocManaged(qc, msize_vol_extra));
   
   memset(*pp, 0, msize_vol_extra);
   memset(*pc, 0, msize_vol_extra);
   memset(*qp, 0, msize_vol_extra);
   memset(*qc, 0, msize_vol_extra);
}
