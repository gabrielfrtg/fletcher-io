#ifndef __HIP_PROPAGATE
#define __HIP_PROPAGATE

void HIP_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it, 
		    float *  pp, float *  pc, float *  qp, float *  qc);

void HIP_SwapArrays(float **pp, float **pc, float **qp, float **qc);

#endif
