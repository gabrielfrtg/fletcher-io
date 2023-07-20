#ifndef _MAP
#define _MAP

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


// mapping 3D array [sz][sy][sx] into 1D [sx*sy*sz] in row-major ordering
// use implicitly requires definition of the variables sx and sy at the place of call


#define ind(ix,iy,iz) (((iz)*sy+(iy))*sx+(ix))


// coord: given i, the map index, return ix, iy, iz


void coord(int i, int sx, int sy, int sz, int *ix, int *iy, int *iz);
#endif
