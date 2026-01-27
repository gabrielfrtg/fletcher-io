#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include"../driver.h"
#include"../sample.h"
#include"hip_stuff.h"
#include"hip_propagate.h"
#include"hip_insertsource.h"
#ifdef USE_HIPCOMP
#include"hip_compression.h"
#endif

// Global device vars
float* dev_pDx=NULL;
float* dev_pDy=NULL;
float* dev_qDx=NULL;
float* dev_qDy=NULL;
float* dev_vpz=NULL;
float* dev_vsv=NULL;
float* dev_epsilon=NULL;
float* dev_delta=NULL;
float* dev_phi=NULL;
float* dev_theta=NULL;
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
                       float *  vpz, float *  vsv, float *  epsilon, float *  delta,
                       float *  phi, float *  theta, 
                       float *  pp, float *  pc, float *  qp, float *  qc)
{

#include"../precomp.h"

	   HIP_Initialize(sx,   sy,   sz,   bord,
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
	HIP_Finalize();
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	HIP_Update_pointers(sx,sy,sz,pc);
}




void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
	              float * pp, float * pc, float * qp, float * qc)
{

	// HIP_Propagate also does TimeForward
	   HIP_Propagate(  sx,   sy,   sz,   bord,
	                    dx,   dy,   dz,   dt,   it,
	                    pp,    pc,    qp,    qc);

}


void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{
	HIP_InsertSource(src, iSource, p, q);
}


#ifdef USE_HIPCOMP
void DRIVER_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                      void** compressed_data, size_t* compressed_size)
{
    HIP_Get_compressed_checkpoint(sx, sy, sz, compressed_data, compressed_size);
}

int DRIVER_Decompress_last(const int sx, const int sy, const int sz,
						   float* host_compressed, size_t compressed_size)
{
	return HIP_Decompress_to_pc(host_compressed, compressed_size, sx, sy, sz);
}

void DRIVER_Decompress_checkpoint_file(const char* infile,
									   const char* out_header,
									   const char* out_data,
									   int sx, int sy, int sz, int bord, int absorb,
									   float dx, float dy, float dz, float dt_output)
{
	HIP_DecompressCheckpointFile(infile, out_header, out_data,
		sx, sy, sz, bord, absorb, dx, dy, dz, dt_output);
}
#endif
