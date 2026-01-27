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

// Global device vars - not needed with unified memory

#define MODEL_GLOBALVARS
#define MODEL_INITIALIZE



void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
                       float dx, float dy, float dz, float dt, float *ch1dxx, float *ch1dyy, float *ch1dzz, float *ch1dxy, float *ch1dyz, float *ch1dxz, float *v2px, float *v2pz, float *v2sz, float *v2pn,
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



void DRIVER_Finalize(const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float *  ch1dxx, float *  ch1dyy, float *  ch1dzz,
               float *  ch1dxy, float *  ch1dyz, float *  ch1dxz,
               float *  v2px, float *  v2pz, float *  v2sz, float *  v2pn,
               float *  vpz, float *  vsv, float *  epsilon, float *  delta,
               float *  phi, float *  theta,
               float *  pp, float *  pc, float *  qp, float *  qc)
{
	HIP_Finalize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc);
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	HIP_Update_pointers(sx,sy,sz,pc);
}




void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it, const float *ch1dxx, const float *ch1dyy, float *ch1dzz, float *ch1dxy, float *ch1dyz, float *ch1dxz, float *v2px, float *v2pz, float *v2sz, float *v2pn,
	              float * pp, float * pc, float * qp, float * qc)
{

	// HIP_Propagate also does TimeForward
	   HIP_Propagate(  sx,   sy,   sz,   bord, dx,   dy,   dz,   dt,   it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp,    pc,    qp,    qc);

}



void DRIVER_InsertSource(float val, int iSource, float *  pc, float *  qc, float *  pp, float *  qp){
	HIP_InsertSource(val, iSource, pc, qc, pp, qp);
}



void DRIVER_Allocate_Model_Variables(float **ch1dxx, float **ch1dyy, float **ch1dzz, float **ch1dxy, float **ch1dyz, float **ch1dxz, float **v2px, float **v2pz, float **v2sz, float **v2pn, int sx, int sy, int sz){
	HIP_Allocate_Model_Variables(ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, sx, sy, sz);
}

void DRIVER_Allocate_main(float **  vpz, float **  vsv, float **  epsilon, float **  delta,
                    float **  phi, float **  theta, float **  pp, float **  pc, float **  qp,
                        float **  qc, int sx, int sy, int sz){

	HIP_Allocate_main(vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc, sx, sy, sz);

}


#ifdef USE_HIPCOMP
void DRIVER_Get_compressed_checkpoint(const int sx, const int sy, const int sz,
                                      float* pc, void** compressed_data, size_t* compressed_size)
{
    HIP_Get_compressed_checkpoint(sx, sy, sz, pc, compressed_data, compressed_size);
}

int DRIVER_Decompress_last(const int sx, const int sy, const int sz,
						   float* host_compressed, size_t compressed_size, float* pc)
{
	return HIP_Decompress_to_pc(host_compressed, compressed_size, sx, sy, sz, pc);
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
