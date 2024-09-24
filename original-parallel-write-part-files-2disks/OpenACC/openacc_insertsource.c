#include "openacc_insertsource.h"


// InsertSource: compute and insert source value at index iSource of arrays p and q


void OPENACC_InsertSource(float dt, int it, int iSource, 
			  float *p, float*q, float src) {

#pragma acc parallel default(present)
  {
     p[iSource]+=src;
     q[iSource]+=src;
  }
}
