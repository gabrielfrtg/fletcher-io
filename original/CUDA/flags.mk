CC=$(GCC)
CFLAGS=-O3
GPUCC=$(NVCC)
GPUCFLAGS=-Xptxas="-v" --maxrregcount 127 --gpu-architecture $(CUDA_GPU_SM)

LIBS =

# NVComp configuration
ifdef USE_NVCOMP
    # NVComp 5.0 requires old ABI
    CFLAGS += -DUSE_NVCOMP -D_GLIBCXX_USE_CXX11_ABI=0
    GPUCFLAGS += -DUSE_NVCOMP -D_GLIBCXX_USE_CXX11_ABI=0
    # Add NVComp paths (expects NVCOMP_ROOT exported, see env.sh)
    NVCOMP_PATH = $(NVCOMP_ROOT)
    CFLAGS += -I$(NVCOMP_PATH)/include -std=c++14
    GPUCFLAGS += -I$(NVCOMP_PATH)/include
    # nvCOMP 5.x provides a single libnvcomp shared object (no libnvcomp_gdeflate)
    LIBS += -L$(NVCOMP_PATH)/lib -lnvcomp -Wl,-rpath,$(NVCOMP_PATH)/lib
endif

# Always append CUDA and standard libs last
LIBS += -L/usr/local/cuda/lib64 -lcudart -lstdc++ $(GCC_LIBS)
