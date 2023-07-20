#ifndef _UTILS
#define _UTILS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "map.h"


// DumpFieldToFile: dumps array into a file using RFS format


void   DumpFieldToFile(int sx, int sy, int sz,
		       int ixStart, int ixEnd,
		       int iyStart, int iyEnd,
		       int izStart, int izEnd,
		       float d1, float d2, float d3,
		       float *arrP, char *fName);


// SlicePtr: contains all info to progressely dump an array that changes over time into an RFS file


#define FNAMEBINARYPATH "./"


enum sliceDirection {XSLICE, YSLICE, ZSLICE, FULL};


typedef struct tsection {
  enum sliceDirection direction;
  int ixStart;
  int ixEnd;
  int iyStart;
  int iyEnd;
  int izStart;
  int izEnd;
  int itCnt;
  float dx;
  float dy;
  float dz;
  float dt;
  FILE *fpHead;
  FILE *fpBinary;
  char fNameHeader[128];
  char fNameBinary[128];
} Slice, *SlicePtr;


// DumpSlicePtr: prints the components of an SlicePtr variable


void DumpSlicePtr(SlicePtr p);


// OpenSliceFile: open file in RFS format that will be continuously appended


SlicePtr OpenSliceFile(int ixStart, int ixEnd,
		       int iyStart, int iyEnd,
		       int izStart, int izEnd,
		       float dx, float dy, float dz, float dt,
		       char *fName);


// DumpSliceFile: appends one array to an opened RFS file 


void DumpSliceFile(int sx, int sy, int sz,
		   float *arrP, SlicePtr p);


// CloseSliceFile: close file in RFS format that has been continuously appended


void CloseSliceFile(SlicePtr p);


// DumpSliceSummary: prints info of one array 


void DumpSliceSummary(int sx, int sy, int sz,
		      SlicePtr p,
		      float dt, int it, float *arrP, float src);


void SwapArrays(float * restrict *pp, float * restrict *pc, float * restrict *qp, float * restrict *qc);
#endif

