#include "openmp_propagate.h"
#include "../derivatives.h"
#include "../map.h"


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time

void OPENMP_Propagate(int sx, int sy, int sz, int bord,
	       float dx, float dy, float dz, float dt, int it, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc) {


#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP


#pragma omp parallel
  { // start omp

    // solve both equations in all internal grid points, 
    // including absortion zone
    
    
#pragma omp for
    for (int iz=bord; iz<sz-bord; iz++) {
      for (int iy=bord; iy<sy-bord; iy++) {
	for (int ix=bord; ix<sx-bord; ix++) {


#define SAMPLE_LOOP
#include "../sample.h"
#undef SAMPLE_LOOP


	}
      }
    }
  } // end omp
}
