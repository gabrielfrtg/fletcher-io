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

#include <time.h>
#include <stdint.h>
#include <inttypes.h>


uint64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}


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
	   float * restrict phi, float * restrict theta, int absorb)
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
  double tdt=0.0;
  uint64_t stamp1 = get_timestamp_ns();

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

      double dd1 = wtime();
      DumpSliceFile_Nofor(sx,sy,sz,pc,sPtr);
      tdt+=wtime()-dd1;
      printf("dump time: %f\n", wtime()-dd1);

      tOut=(++nOut)*dtOutput;
#ifdef _DUMP
      DRIVER_Update_pointers(sx,sy,sz,pc);
      //      DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,src);
#endif
    }
  }

  uint64_t stamp2 = get_timestamp_ns();

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

	double execution_time = ((double)(stamp2-stamp1))*1e-9;
  
  printf("Total dump time (s): %f\n", tdt);
  printf ("Execution time (s) is %lf\n", walltime);
  printf ("Total execution time (s) is %lf\n", execution_time);
  printf ("MSamples/s %.0lf\n", MSamples);
  printf ("Memory High Water Mark is %ld %s\n",HWM, HWMUnit);

  printf("original,%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%f,%f,%lu,%lu,%lf,%lf,%.0lf\n", 
          sPtr->fName, sx - 2*bord - 2*absorb, sy - 2*bord - 2*absorb, sz - 2*bord - 2*absorb, absorb, dx, dy, dz, dt, st*dt, 
          stamp1, stamp2, walltime, execution_time, MSamples);

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

