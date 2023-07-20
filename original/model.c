#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"
#ifdef PAPI
#include "ModPAPI.h"
#endif

#define MODEL_GLOBALVARS
#include "precomp.h"
#undef MODEL_GLOBALVARS


void ReportProblemSizeCSV(const int sx, const int sy, const int sz,
			  const int bord, const int st, 
			  FILE *f){
  fprintf(f,
	  "sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",
	  sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,
		      long HWM, char *HWMUnit, FILE *f){
  fprintf(f,
	  "walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",
	  walltime, MSamples, HWM, HWMUnit);
}


void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr, 
           const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it, 
	   float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc,
	   float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	   float * restrict phi, float * restrict theta)
{

  float tSim=0.0;
  int nOut=1;
  float tOut=nOut*dtOutput;

  const long samplesPropagate=(long)(sx-2*bord)*(long)(sy-2*bord)*(long)(sz-2*bord);
  const long totalSamples=samplesPropagate*(long)st;

#ifdef PAPI
  long long values[NCOUNTERS];
  long long ThisValues[NCOUNTERS];
  for (int i=0; i<NCOUNTERS; i++) {
    values[i]=0LL;
    ThisValues[i]=0LL;
  }

  const int eventset=InitPAPI_CreateCounters();
#endif

#define MODEL_INITIALIZE
#include "precomp.h"
#undef MODEL_INITIALIZE

  // DRIVER_Initialize initialize target, allocate data etc
  DRIVER_Initialize(sx,   sy,   sz,   bord,
		      dx,  dy,  dz,  dt,
		      vpz,    vsv,    epsilon,    delta,
		      phi,    theta,
		      pp,    pc,    qp,    qc);

  
  double walltime=0.0;
  double tt1 = wtime();

  for (int it=1; it<=st; it++) {

    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    
    DRIVER_InsertSource(dt,it-1,iSource,pc,qc,src);

#ifdef PAPI
    StartCounters(eventset);
#endif

    const double t0=wtime();
    DRIVER_Propagate(  sx,   sy,   sz,   bord,
		       dx,   dy,   dz,   dt,   it,
		       pp,    pc,    qp,    qc);

    SwapArrays(&pp, &pc, &qp, &qc);
    walltime+=wtime()-t0;

#ifdef PAPI
    StopReadCounters(eventset, ThisValues);
    for (int i=0; i<NCOUNTERS; i++) {
      values[i]+=ThisValues[i];
    }
#endif

    tSim=it*dt;
    if (tSim >= tOut) {

      DRIVER_Update_pointers(sx,sy,sz,pc);
      DumpSliceFile(sx,sy,sz,pc,sPtr);
      tOut=(++nOut)*dtOutput;
#ifdef _DUMP
      DRIVER_Update_pointers(sx,sy,sz,pc);
      //      DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,src);
#endif
    }
  }

  double tt2 = wtime();

  // get HWM data

#define MEGA 1.0e-6
#define GIGA 1.0e-9
  const char StringHWM[6]="VmHWM";
  char line[256], title[12],HWMUnit[8];
  const long HWM;
  const double MSamples=(MEGA*(double)totalSamples)/walltime;
  
  FILE *fp=fopen("/proc/self/status","r");
  while (fgets(line, 256, fp) != NULL){
    if (strncmp(line, StringHWM, 5) == 0) {
      sscanf(line+6,"%ld %s", &HWM, HWMUnit);
      break;
    }
  }
  fclose(fp);

  // Dump Execution Metrics
  
  printf ("Execution time (s) is %lf\n", walltime);
  printf ("Total execution time (s) is %lf\n", tt2 - tt1);
  printf ("MSamples/s %.0lf\n", MSamples);
  printf ("Memory High Water Mark is %ld %s\n",HWM, HWMUnit);

  // Dump Execution Metrics in CSV
  
  FILE *fr=NULL;
  const char fName[]="Report.csv";
  fr=fopen(fName,"w");

  // report problem size

  ReportProblemSizeCSV(sx, sy, sz,
		       bord, st, 
		       fr);

  // report collected metrics

  ReportMetricsCSV(walltime, MSamples,
		   HWM, HWMUnit, fr);
  
  // report PAPI metrics

#ifdef PAPI
  ReportRawCountersCSV (values, fr);
#endif
  
  fclose(fr);

  fflush(stdout);

  // DRIVER_Finalize deallocate data, clean-up things etc 
  DRIVER_Finalize();

}

