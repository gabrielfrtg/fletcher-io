#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "boundary.h"
#include "source.h"
#include "utils.h"
#include "map.h"
#include "driver.h"
#include "fletcher.h"
#include "model.h"

enum Form {ISO, VTI, TTI};

int main(int argc, char** argv) {

  enum Form prob;        // problem formulation
  int nx;                // grid points in x
  int ny;                // grid points in y
  int nz;                // grid points in z
  int bord=4;            // border size to apply the stencil at grid extremes
  int absorb;            // absortion zone size
  int sx;                // grid dimension in x (grid points + 2*border + 2*absortion)
  int sy;                // grid dimension in y (grid points + 2*border + 2*absortion)
  int sz;                // grid dimension in z (grid points + 2*border + 2*absortion)
  int st;                // number of time steps
  float dx;              // grid step in x
  float dy;              // grid step in y
  float dz;              // grid step in z
  float dt;              // time advance at each time step
  float tmax;            // desired simulation final time
  int ixSource;          // source x index
  int iySource;          // source y index
  int izSource;          // source z index
  int iSource;           // source index (ix,iy,iz) maped into 1D array
//PPL  int i, ix, iy, iz, it; // for indices
  int i, it;             // for indices
//PPL  char fNameAbs[128];    // prefix of absortion file
  char fNameSec[128];    // prefix of sections files

  const float dtOutput=0.01;

  it = 0; //PPL
    
  // input problem definition
  
  if (argc<ARGS) {
    printf("program requires %d input arguments; execution halted\n",ARGS-1);
    exit(-1);
  } 
  strcpy(fNameSec,argv[1]);
  nx=atoi(argv[2]);
  ny=atoi(argv[3]);
  nz=atoi(argv[4]);
  absorb=atoi(argv[5]);
  dx=atof(argv[6]);
  dy=atof(argv[7]);
  dz=atof(argv[8]);
  dt=atof(argv[9]);
  tmax=atof(argv[10]);

  // verify problem formulation

  if (strcmp(fNameSec,"ISO")==0) {
    prob=ISO;
  }
  else if (strcmp(fNameSec,"VTI")==0) {
    prob=VTI;
  }
  else if (strcmp(fNameSec,"TTI")==0) {
    prob=TTI;
  }
  else {
    printf("Input problem formulation (%s) is unknown\n", fNameSec);
    exit(-1);
  }

#ifdef _DUMP
  printf("Problem is ");
  switch (prob) {
  case ISO:
    printf("isotropic\n");
    break;
  case VTI:
    printf("anisotropic with vertical transversely isotropy using sigma=%f\n", SIGMA);
    break;
  case TTI:
    printf("anisotropic with tilted transversely isotropy using sigma=%f\n", SIGMA);
    break;
  }
#endif

  // grid dimensions from problem size

  sx=nx+2*bord+2*absorb;
  sy=ny+2*bord+2*absorb;
  sz=nz+2*bord+2*absorb;

  // number of time iterations

  st=ceil(tmax/dt);

  // source position

  ixSource=sx/2;
  iySource=sy/2;
  izSource=sz/2;
  iSource=ind(ixSource,iySource,izSource);

  // dump problem input data

#ifdef _DUMP
  printf("Grid size is (%d,%d,%d) with spacing (%.2f,%.2f,%.2f); simulated area (%.2f,%.2f,%.2f) \n", 
	 nx, ny, nz, dx, dy, dz, (nx-1)*dx, (ny-1)*dy, (nz-1)*dz);
  printf("Grid is extended by %d absortion points and %d border points at each extreme\n", absorb, bord);
  printf("Wave is propagated at internal+absortion points of size (%d,%d,%d)\n",
	 nx+2*absorb, ny+2*absorb, nz+2*absorb);
  printf("Source at coordinates (%d,%d,%d)\n", ixSource,iySource,izSource);
  printf("Will run %d time steps of %f to reach time %f\n", st, dt, st*dt);

#ifdef _OPENMP
#include <omp.h>
#pragma omp parallel
  if (omp_get_thread_num() == 0) {
    if (omp_get_num_threads()==1)
      printf("Sequential execution with OpenMP directives enable\n");
    else
      printf("Execution with OpenMP directives on %d threads\n", omp_get_num_threads());
  }
#else
  printf("OpenMP directives disabled\n");
#endif

#ifdef _OPENACC
#include <openacc.h>
  const acc_device_t devType=acc_get_device_type();
  const int devNum=acc_get_device_num(devType);
  const char* propName=acc_get_property_string(devNum, devType, acc_property_name);
  printf("Execution with OpenACC directives on device %s\n",propName);
#else
  printf("OpenACC directives disabled\n");
#endif

#endif

  // allocate input anisotropy arrays
  
  float *vpz=NULL;      // p wave speed normal to the simetry plane
  vpz = (float *) malloc(sx*sy*sz*sizeof(float));

  float *vsv=NULL;      // sv wave speed normal to the simetry plane
  vsv = (float *) malloc(sx*sy*sz*sizeof(float));
  
  float *epsilon=NULL;  // Thomsen isotropic parameter
  epsilon = (float *) malloc(sx*sy*sz*sizeof(float));
  
  float *delta=NULL;    // Thomsen isotropic parameter
  delta = (float *) malloc(sx*sy*sz*sizeof(float));
  
  float *phi=NULL;     // isotropy simetry azimuth angle
  phi = (float *) malloc(sx*sy*sz*sizeof(float));
  
  float *theta=NULL;  // isotropy simetry deep angle
  theta = (float *) malloc(sx*sy*sz*sizeof(float));

  // input anisotropy arrays for selected problem formulation

  switch(prob) {

  case ISO:

    for (i=0; i<sx*sy*sz; i++) {
      vpz[i]=3000.0;
      epsilon[i]=0.0;
      delta[i]=0.0;
      phi[i]=0.0;
      theta[i]=0.0;
      vsv[i]=0.0;
    }
    break;

  case VTI:

    if (SIGMA > MAX_SIGMA) {
      printf("Since sigma (%f) is greater that threshold (%f), sigma is considered infinity and vsv is set to zero\n", 
		      SIGMA, MAX_SIGMA);
    }
    for (i=0; i<sx*sy*sz; i++) {
      vpz[i]=3000.0;
      epsilon[i]=0.24;
      delta[i]=0.1;
      phi[i]=0.0;
      theta[i]=0.0;
      if (SIGMA > MAX_SIGMA) {
	vsv[i]=0.0;
      } else {
	vsv[i]=vpz[i]*sqrtf(fabsf(epsilon[i]-delta[i])/SIGMA);
      }
    }
    break;

  case TTI:

    if (SIGMA > MAX_SIGMA) {
      printf("Since sigma (%f) is greater that threshold (%f), sigma is considered infinity and vsv is set to zero\n", 
		      SIGMA, MAX_SIGMA);
    }
    for (i=0; i<sx*sy*sz; i++) {
      vpz[i]=3000.0;
      epsilon[i]=0.24;
      delta[i]=0.1;
      //      phi[i]=0.0;
      phi[i]=1.0; // evitando coeficientes nulos
      theta[i]=atanf(1.0);
      if (SIGMA > MAX_SIGMA) {
	vsv[i]=0.0;
      } else {
	vsv[i]=vpz[i]*sqrtf(fabsf(epsilon[i]-delta[i])/SIGMA);
      }
    }
  } // end switch

  // stability condition
  
  float maxvel;
  maxvel=vpz[0]*sqrt(1.0+2*epsilon[0]);
  for (i=1; i<sx*sy*sz; i++) {
    maxvel=fmaxf(maxvel,vpz[i]*sqrt(1.0+2*epsilon[i]));
  }
  float mindelta=dx;
  if (dy<mindelta)
    mindelta=dy;
  if (dz<mindelta)
    mindelta=dz;
  float recdt;
  recdt=(MI*mindelta)/maxvel;
#ifdef _DUMP
  printf("Recomended maximum time step is %f; used time step is %f\n", recdt, dt);
#endif

  // random boundary speed

  RandomVelocityBoundary(sx, sy, sz,
			 nx, ny, nz,
			 bord, absorb,
			 vpz, vsv);
  // pressure fields at previous, current and future time steps
  
  float *pp=NULL;
  pp = (float *) malloc(sx*sy*sz*sizeof(float)); 
  float *pc=NULL;
  pc = (float *) malloc(sx*sy*sz*sizeof(float)); 
  float *qp=NULL;
  qp = (float *) malloc(sx*sy*sz*sizeof(float)); 
  float *qc=NULL;
  qc = (float *) malloc(sx*sy*sz*sizeof(float)); 
  for (i=0; i<sx*sy*sz; i++) {
    pp[i]=0.0f; pc[i]=0.0f; 
    qp[i]=0.0f; qc[i]=0.0f;
  }

  // slices

//PPL  char fName[10];
  int ixStart=0;
  int ixEnd=sx-1;
  int iyStart=0;
  int iyEnd=sy-1;
  int izStart=0;
  int izEnd=sz-1;

  SlicePtr sPtr;
  sPtr=OpenSliceFile(ixStart, ixEnd,
		     iyStart, iyEnd,
		     izStart, izEnd,
		     dx, dy, dz, dt,
		     fNameSec);

  DumpSliceFile(sx,sy,sz,pc,sPtr);
#ifdef _DUMP
  DumpSlicePtr(sPtr);
  //  DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,0);
#endif
  
  // Model do:
  // - Initialize
  // - time loop
  // - calls Propagate
  // - calls TimeForward
  // - calls InsertSource
  // - do AbsorbingBoundary and DumpSliceFile, if needed
  // - Finalize
  Model(st,     iSource, dtOutput, sPtr,
        sx,     sy,      sz,       bord,
        dx,     dy,      dz,       dt,   it, 
        pp,     pc,      qp,       qc,
	vpz,    vsv,     epsilon,  delta,
	phi,    theta, absorb);
}
