#include "ModPAPI.h"

// PAPIFails:
//   convert PAPI return code on failure and stops


void PAPIFails(char *s, int retval) {
  printf("**(PAPIFails)**: PAPI funcion %s failed with retval=%d meaning %s\n",
	 s, retval, PAPI_strerror(retval));
  exit(-1);
} 


// InitPAPI_CreateCounters:
//    initialize PAPI library and create the counters to measure
//    memory hierarchy usage
// Returns eventset, used by other PAPI routines


int InitPAPI_CreateCounters(){
  int eventset=PAPI_NULL;
  int retval;
  
  /* Init the PAPI library */
  retval = PAPI_library_init( PAPI_VER_CURRENT );
  if ( retval != PAPI_VER_CURRENT ) {
    PAPIFails("PAPI_library_init", retval );
  }
  
  /* Create the eventset */
  retval=PAPI_create_eventset(&eventset);
  if (retval!=PAPI_OK) {
    PAPIFails("PAPI_create_eventset", retval );
  }

  int ic, il;
  for (int i=0; i<NCOUNTERS; i++) {
    retval=PAPI_add_named_event(eventset,eventName[i]); 
    if (retval!=PAPI_OK) {
      char msg[256]="PAPI_add_named_event("; 
      for (ic=0; ic<256; ic++) {
	if (msg[ic]=='\0') {
	  break;
	}
      }
      for (il=0; il<16; il++, ic++) {
	msg[ic]=eventName[i][il];
	if (eventName[i][il] =='\0') {
	  msg[ic]=')';
	  msg[ic+1]='\0';
	  break;
	}
      }
      printf("msg=%s\n",msg);
      PAPIFails(msg, retval );
    }
  }
  return(eventset);
}


// StartCounters:
//    reset and start counting the events on eventset


void StartCounters(int eventset){
    PAPI_start(eventset);
}


// StopCounters:
//    stop counting the events on eventset and return their values


void StopReadCounters(int eventset, long long *val) {
  PAPI_stop(eventset, val);
}


// ReportRawCountersCSV
//    report raw counter names and values at a file with CSV format
//    do not report if *f is NULL


void ReportRawCountersCSV (long long *values, FILE *f){

  if (f != NULL) {
    for (int i=0; i<NCOUNTERS; i++) {
      fprintf(f,"%s; %lld; ",eventName[i], values[i]);
    }
    fprintf(f,"\n");
  }
}

