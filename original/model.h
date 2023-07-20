#ifndef _MODEL
#define _MODEL

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr, 
           const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it, 
	   float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc,
	   float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	   float * restrict phi, float * restrict theta);

#endif
