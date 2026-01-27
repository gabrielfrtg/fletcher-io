#include "hip/hip_runtime.h"
#include "hip_defines.h"
#include "hip_insertsource.h"

extern int BSIZE_X;

__launch_bounds__(64)
__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[iSource]+=val;
    qc[iSource]+=val;
  }
}


void HIP_InsertSource(const float val, const int iSource, float *p, float *q, float *pp, float *qp)
{

	
  if ((pp) && (qp))
  {
     dim3 threadsPerBlock(1, 1);
     dim3 numBlocks(1,1);
  
//     kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, dev_pc, dev_qc);
     hipLaunchKernelGGL(kernel_InsertSource, numBlocks, threadsPerBlock, 0, 0, val, iSource, p, q);
     HIP_CALL(hipGetLastError());
     HIP_CALL(hipDeviceSynchronize());
  }
}
