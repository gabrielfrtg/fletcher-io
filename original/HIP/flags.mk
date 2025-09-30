CC=$(GCC)
CFLAGS=-O3 -fopenmp
GPUCC=$(HIPCC)
GPUCFLAGS=-fPIE
LIBS=-L/opt/rocm/hip/lib -lamdhip64 $(GCC_LIBS)
