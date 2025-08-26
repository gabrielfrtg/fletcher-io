CC=$(GCC)
CFLAGS=-O3
GPUCC=$(NVCC)
GPUCFLAGS=-Xptxas="-v" --maxrregcount 127 --gpu-architecture $(CUDA_GPU_SM)

LIBS =

# NVComp configuration
ifdef USE_NVCOMP
    NVCOMP_PATH = $(NVCOMP_ROOT)
    CFLAGS += -I$(NVCOMP_PATH)/include -DUSE_NVCOMP
    GPUCFLAGS += -I$(NVCOMP_PATH)/include -DUSE_NVCOMP
    LIBS += -L$(NVCOMP_PATH)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_PATH)/lib
endif

LIBS += -L/usr/local/cuda/lib64 -lcudart -lstdc++ $(GCC_LIBS)
