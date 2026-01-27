#ifndef __HIP_PROPAGATE
#define __HIP_PROPAGATE

void HIP_Propagate(const int sx, const int sy, const int sz, const int bord, const float dx, const float dy, const float dz, const float dt, const int it, const float *ch1dxx, const float *ch1dyy, float *ch1dzz, float *ch1dxy, float *ch1dyz, float *ch1dxz, float *v2px, float *v2pz, float *v2sz, float *v2pn, float *  pp, float *  pc, float *  qp, float *  qc);

void HIP_SwapArrays(float **pp, float **pc, float **qp, float **qc);

#endif
