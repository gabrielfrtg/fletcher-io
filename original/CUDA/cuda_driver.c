#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include"../driver.h"
#include"../sample.h"
#include"cuda_stuff.h"
#include"cuda_propagate.h"
#include"cuda_insertsource.h"

// Global device vars
float* dev_pDx=NULL;
float* dev_pDy=NULL;
float* dev_qDx=NULL;
float* dev_qDy=NULL;
float* dev_ch1dxx=NULL;
float* dev_ch1dyy=NULL;
float* dev_ch1dzz=NULL;
float* dev_ch1dxy=NULL;
float* dev_ch1dyz=NULL;
float* dev_ch1dxz=NULL;
float* dev_v2px=NULL;
float* dev_v2pz=NULL;
float* dev_v2sz=NULL;
float* dev_v2pn=NULL;
float* dev_pp=NULL;
float* dev_pc=NULL;
float* dev_qp=NULL;
float* dev_qc=NULL;


#define MODEL_GLOBALVARS
#define MODEL_INITIALIZE



void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
                       float dx, float dy, float dz, float dt,
                       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
                       float * restrict phi, float * restrict theta, 
                       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

#include"../precomp.h"

	   CUDA_Initialize(sx,   sy,   sz,   bord,
		  dx,  dy,  dz,  dt,
	          ch1dxx,    ch1dyy,    ch1dzz, 
  	          ch1dxy,    ch1dyz,    ch1dxz, 
  	          v2px,    v2pz,    v2sz,    v2pn,
  	          vpz,    vsv,    epsilon,    delta,
  	          phi,    theta,
  	          pp,    pc,    qp,    qc);

}



void DRIVER_Finalize()
{
	CUDA_Finalize();
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	CUDA_Update_pointers(sx,sy,sz,pc);
}




void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
	              float * pp, float * pc, float * qp, float * qc)
{

	// CUDA_Propagate also does TimeForward
	   CUDA_Propagate(  sx,   sy,   sz,   bord,
	                    dx,   dy,   dz,   dt,   it,
	                    pp,    pc,    qp,    qc);

}


void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{
	CUDA_InsertSource(src, iSource, p, q);
}

