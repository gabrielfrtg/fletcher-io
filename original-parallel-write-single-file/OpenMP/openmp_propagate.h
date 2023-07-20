#ifndef _OPENMP_PROPAGATE
#define _OPENMP_PROPAGATE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time


void OPENMP_Propagate(int sx, int sy, int sz, int bord,
	       float dx, float dy, float dz, float dt, int it, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);

#endif
