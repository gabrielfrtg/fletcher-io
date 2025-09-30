CC=$(GCC)
CFLAGS=-O3 -fopenmp
GPUCC=$(HIPCC)
GPUCFLAGS=-fPIE
LIBS=-L/opt/rocm/hip/lib -lamdhip64 $(GCC_LIBS)

ifdef USE_HIPCOMP
    HIPCOMP_PATH = $(HIPCOMP_ROOT)
    CFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    GPUCFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    LIBS += -L$(HIPCOMP_PATH)/lib -lhipcomp -Wl,-rpath,$(HIPCOMP_PATH)/lib
endif
