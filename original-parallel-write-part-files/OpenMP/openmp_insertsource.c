#include "openmp_insertsource.h"


// InsertSource: compute and insert source value at index iSource of arrays p and q


void OPENMP_InsertSource(float dt, int it, int iSource, 
			  float *p, float*q, float src) {
  {
     p[iSource]+=src;
     q[iSource]+=src;
  }
}
