#ifndef __HIP_STUFF
#define __HIP_STUFF

#ifdef __cplusplus
extern "C" {
#endif

void HIP_Initialize(const int sx, const int sy, const int sz, const int bord,
	       float dx, float dy, float dz, float dt,
	       float *  ch1dxx, float *  ch1dyy, float *  ch1dzz, 
	       float *  ch1dxy, float *  ch1dyz, float *  ch1dxz, 
	       float *  v2px, float *  v2pz, float *  v2sz, float *  v2pn,
	       float *  vpz, float *  vsv, float *  epsilon, float *  delta,
	       float *  phi, float *  theta, 
	       float *  pp, float *  pc, float *  qp, float *  qc);

void HIP_Finalize();

void HIP_Update_pointers(const int sx, const int sy, const int sz, float *pc);

#ifdef __cplusplus
}
#endif

#endif
