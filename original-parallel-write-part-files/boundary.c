#include "boundary.h"


// RandomVelocityBoundary: creates a boundary with random velocity around domain


void RandomVelocityBoundary(int sx, int sy, int sz,
			    int nx, int ny, int nz,
			    int bord, int absorb,
			    float *vpz, float *vsv) {

  int i, ix, iy, iz;
  int distx, disty, distz, dist;
  int ivelx, ively, ivelz;
  float bordDist;
  float frac, rfac;
  int firstIn, bordLen;
  float maxP, maxS;

  // maximum speed of P and S within bounds
  maxP=0.0; maxS=0.0;
  for (iz=bord+absorb; iz<nz+bord+absorb; iz++) {
    for (iy=bord+absorb; iy<ny+bord+absorb; iy++) {
      for (i=ind(bord+absorb,iy,iz); i<ind(nx+bord+absorb,iy,iz); i++) {
	maxP=fmaxf(vpz[i],maxP);
	maxS=fmaxf(vsv[i],maxS);
      }
    }
  }
    
  bordLen=bord+absorb-1;   // last index on low absortion zone
  firstIn=bordLen+1;       // first index inside input grid 
  frac=1.0/(float)(absorb);

  for (iz=0; iz<sz; iz++) {
    for (iy=0; iy<sy; iy++) {
      for (ix=0; ix<sx; ix++) {
	i=ind(ix,iy,iz);
	// do nothing inside input grid
	if ((iz>=firstIn && iz<=bordLen+nz) &&
	    (iy>=firstIn && iy<=bordLen+ny) &&
	    (ix>=firstIn && ix<=bordLen+nx)) {
	  continue;
	}
	// random speed inside absortion zone
	else if ((iz>=bord && iz<=2*bordLen+nz) &&
		 (iy>=bord && iy<=2*bordLen+ny) &&
		 (ix>=bord && ix<=2*bordLen+nx)) {
	  if (iz>bordLen+nz) {
	    distz=iz-bordLen-nz;
	    ivelz=bordLen+nz;
	  } else if (iz<firstIn) {
	    distz=firstIn-iz;
	    ivelz=firstIn;
	  } else {
	    distz=0;
	    ivelz=iz;
	  }
	  if (iy>bordLen+ny) {
	    disty=iy-bordLen-ny;
	    ively=bordLen+ny;
	  } else if (iy<firstIn) {
	    disty=firstIn-iy;
	    ively=firstIn;
	  } else {
	    disty=0;
	    ively=iy;
	  }
	  if (ix>bordLen+nx) {
	    distx=ix-bordLen-nx;
	    ivelx=bordLen+nx;
	  } else if (ix<firstIn) {
	    distx=firstIn-ix;
	    ivelx=firstIn;
	  } else {
	    distx=0;
	    ivelx=ix;
	  }
	  dist=(disty>distz)?disty:distz;
	  dist=(dist >distx)?dist :distx;
	  bordDist=(float)(dist)*frac;
	  rfac=(float)rand()/(float)RAND_MAX;
	  vpz[i]=vpz[ind(ivelx,ively,ivelz)]*(1.0-bordDist)+
	    maxP*rfac*bordDist;
	  vsv[i]=vsv[ind(ivelx,ively,ivelz)]*(1.0-bordDist)+
	    maxS*rfac*bordDist;
	}
	// null speed at border
	else
	//PPL added {} surrounding vpz and vsv lines below
	{
	  vpz[i]=0.0;
	  vsv[i]=0.0;
	}
      }
    }
  }
}
