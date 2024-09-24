#include<stdio.h>
#include<stdlib.h>
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define ERRO(str) do{ fprintf(stderr, str "\n"); exit(1); }while(0)

// A really small number 
#define EPS 1.0e-15
//#define MAX_ERROR 1.0e-5
#define MAX_ERROR 1.0e-5

int main(int argc, char *argv[])
{
   if (argc != 8)
   {
      fprintf(stderr, "Use %s n1 n2 n3 n4 reference_file new_result output_diff\n", argv[0]);
      return 1;
   }

   const long n1=atol(argv[1]);
   const long n2=atol(argv[2]);
   const long n3=atol(argv[3]);
   const long n4=atol(argv[4]);

   const char * const file1 = argv[5];
   const char * const file2 = argv[6];
   const char * const file3 = argv[7];

   printf("argc=%d\n", argc);
   printf("n1=%ld n2=%ld n3=%ld n4=%ld\n", n1, n2, n3, n4);
   printf("file1: %s\n", file1);
   printf("file2: %s\n", file2);
   printf("file3: %s\n", file3);

   const size_t msize=n1*n2*sizeof(float);
   float *plane1 = malloc(n1*n2*sizeof(float));
   float *plane2 = malloc(n1*n2*sizeof(float));
   float *diff   = malloc(n1*n2*sizeof(float));
   FILE* f1 = fopen(file1, "r");
   if (f1 == NULL) ERRO("fopen");
   FILE* f2 = fopen(file2, "r");
   if (f2 == NULL) ERRO("fopen");
   FILE* f3 = fopen(file3, "w");
   if (f3 == NULL) ERRO("fopen");
  
  
   long global_cont=0;
   for (int it=0; it<n4; it++)
   {
      float maxval=EPS; // per timestep maxval for error detection
      for (int iz=0; iz<n3; iz++)
      {
          if (fread(plane1, msize, (size_t) 1, f1) != 1) ERRO("fread");
          if (fread(plane2, msize, (size_t) 1, f2) != 1) ERRO("fread");
          for (int i=0; i<n1*n2; i++) maxval=MAX(maxval, fabsf(plane1[i]));
          long cont=0;
          for (int i=0; i<n1*n2; i++)
          {
             diff[i]=plane1[i]-plane2[i];
             if (fabsf(diff[i]) > (maxval*MAX_ERROR)) cont++;
          }
          global_cont += cont;
          if (cont) printf("%ld erros no plano it=%d iz=%d maxval=%lf\n", cont, it, iz, maxval);
          if (fwrite(diff, msize, (size_t) 1, f3) != 1) ERRO("fwrite");
      }
      printf("it=%d maxval=%lf\n", it, maxval);
   }
   fclose(f1);
   fclose(f2);
   fclose(f3);

   if (global_cont) return 1;
   printf("success\n");
   return 0;
}
