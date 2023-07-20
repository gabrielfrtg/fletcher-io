#include "map.h"

 
// coord: given i, the map index, return ix, iy, iz


void coord(int i, int sx, int sy, int sz, int *ix, int *iy, int *iz) {
  int rem;
  *ix=i%sx;
  rem=(i-*ix)/sx;
  *iy=rem%sy;
  *iz=(rem-*iy)/sy;
}
