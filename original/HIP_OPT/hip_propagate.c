#include "hip/hip_runtime.h"
#include "hip_defines.h"
#include "hip_propagate.h"
#include "../derivatives.h"
#include "../map.h"

extern int BSIZE_X, BSIZE_Y, BSIZE_Z;

__launch_bounds__(256)
__global__ void kernel_Propagate(const int sx, const int sy, const int sz, const int bord,
				 const float dx, const float dy, const float dz, const float dt,  
				 const int it, const float * const  ch1dxx, 
				 const float * const  ch1dyy, float *  ch1dzz, 
				 float *  ch1dxy, float *  ch1dyz, float *  ch1dxz, 
				 float *  v2px, float *  v2pz, float *  v2sz, 
				 float *  v2pn, float *  pp, float *  pc, 
				 float *  qp, float *  qc)
{
	const int ix=blockIdx.x * blockDim.x + threadIdx.x;
	const int iy=blockIdx.y * blockDim.y + threadIdx.y;
	const int iz=blockIdx.z * blockDim.z + threadIdx.z;

	const int strideX=ind(1,0,0)-ind(0,0,0);
	const int strideY=ind(0,1,0)-ind(0,0,0);
	const int strideZ=ind(0,0,1)-ind(0,0,0);
	const float dxxinv=1.0f/(dx*dx);
	const float dyyinv=1.0f/(dy*dy);
	const float dzzinv=1.0f/(dz*dz);
	const float dxyinv=1.0f/(dx*dy);
	const float dxzinv=1.0f/(dx*dz);
	const float dyzinv=1.0f/(dy*dz);

	// solve both equations in all internal grid points, 
    	// including absortion zone
	if(ix < bord || ix > sx-bord || iy < bord || iy > sy-bord || iz < bord || iz > sz-bord)
		return;
	const int i=ind(ix,iy,iz);
		// p derivatives, H1(p) and H2(p)
	const float pxx= Der2(pc, i, strideX, dxxinv);
	const float pyy= Der2(pc, i, strideY, dyyinv);
	const float pzz= Der2(pc, i, strideZ, dzzinv);
	const float pxy= DerCross(pc, i, strideX, strideY, dxyinv);
	const float pyz= DerCross(pc, i, strideY, strideZ, dyzinv);
	const float pxz= DerCross(pc, i, strideX, strideZ, dxzinv);
	const float cpxx=ch1dxx[i]*pxx;
	const float cpyy=ch1dyy[i]*pyy;
	const float cpzz=ch1dzz[i]*pzz;
	const float cpxy=ch1dxy[i]*pxy;
	const float cpxz=ch1dxz[i]*pxz;
	const float cpyz=ch1dyz[i]*pyz;
	const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
	const float h2p=pxx+pyy+pzz-h1p;
	// q derivatives, H1(q) and H2(q)
	const float qxx= Der2(qc, i, strideX, dxxinv);
	const float qyy= Der2(qc, i, strideY, dyyinv);
	const float qzz= Der2(qc, i, strideZ, dzzinv);
	const float qxy= DerCross(qc, i, strideX,  strideY, dxyinv);
	const float qyz= DerCross(qc, i, strideY,  strideZ, dyzinv);
	const float qxz= DerCross(qc, i, strideX,  strideZ, dxzinv);
	const float cqxx=ch1dxx[i]*qxx;
	const float cqyy=ch1dyy[i]*qyy;
	const float cqzz=ch1dzz[i]*qzz;
	const float cqxy=ch1dxy[i]*qxy;
	const float cqxz=ch1dxz[i]*qxz;
	const float cqyz=ch1dyz[i]*qyz;
	const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
	const float h2q=qxx+qyy+qzz-h1q;
	// p-q derivatives, H1(p-q) and H2(p-q)
	const float h1pmq=h1p-h1q;
	const float h2pmq=h2p-h2q;
	// rhs of p and q equations
	const float rhsp=v2px[i]*h2p + v2pz[i]*h1q + v2sz[i]*h1pmq;
	const float rhsq=v2pn[i]*h2p + v2pz[i]*h1q - v2sz[i]*h2pmq;
	// new p and q
	pp[i]=2.0f*pc[i] - pp[i] + rhsp*dt*dt;
	qp[i]=2.0f*qc[i] - qp[i] + rhsq*dt*dt;
 	
}


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void HIP_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it, 
		    float *  pp, float *  pc, float *  qp, float *  qc)
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
  
  dim3 threadsPerBlock(BSIZE_X,BSIZE_Y,BSIZE_Z);
  dim3 numBlocks((sx-2*bord + threadsPerBlock.x - 1) / threadsPerBlock.x, (sy-2*bord + threadsPerBlock.y - 1) / threadsPerBlock.y,(sz-2*bord + threadsPerBlock.z - 1) / threadsPerBlock.z);
  //dim3 numBlocks((sx-1)/threadsPerBlock.x+1, (sy-1)/threadsPerBlock.y + 1, (sz -1)/threadsPerBlock.z + 1);
  
  hipLaunchKernelGGL(kernel_Propagate, numBlocks, threadsPerBlock, 0, 0, sx, sy, sz, bord,
						      dx,   dy,   dz,   dt,   it, 
						      dev_ch1dxx,  dev_ch1dyy,  dev_ch1dzz, 
						      dev_ch1dxy,  dev_ch1dyz,  dev_ch1dxz, 
						      dev_v2px,  dev_v2pz,  dev_v2sz,  dev_v2pn,
						      dev_pp,  dev_pc,  dev_qp,  dev_qc);
  HIP_CALL(hipGetLastError());
  HIP_SwapArrays(&dev_pp, &dev_pc, &dev_qp, &dev_qc);
  HIP_CALL(hipDeviceSynchronize());
}

// swap array pointers on time forward array propagation
void HIP_SwapArrays(float **pp, float **pc, float **qp, float **qc) {
  float *tmp;
  
  tmp=*pp;
  *pp=*pc;
  *pc=tmp;
  
  tmp=*qp;
  *qp=*qc;
  *qc=tmp;
}
