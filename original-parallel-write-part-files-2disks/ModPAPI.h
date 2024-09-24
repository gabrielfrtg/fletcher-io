#ifndef _MODPAPI
#define _MODPAPI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"

// vector instruction length
#define VLEN 4LL

// PAPI counters and counter names
//#define NCOUNTERS 4
//static char eventName[NCOUNTERS][16]={"PAPI_LD_INS", "PAPI_L1_LDM", "PAPI_L2_LDM", "PAPI_L3_LDM"};

#define NCOUNTERS 1
static char eventName[NCOUNTERS][16]={"PAPI_SP_OPS"};

// InitPAPI_CreateCounters:
//    initialize PAPI library and create the counters to measure
//    memory hierarchy usage
// Returns eventset, used by other PAPI routines

int InitPAPI_CreateCounters();

// StartCounters:
//    reset and start counting the events on eventset

void StartCounters(int eventset);

// StopCounters:
//    stop counting the events on eventset and return their values

void StopReadCounters(int eventset, long long *val);

// ReportRawCountersCSV
//    report raw counter names and values at a file with CSV format
//    do not report if *f is NULL

void ReportRawCountersCSV (long long *values, FILE *f);
#endif
