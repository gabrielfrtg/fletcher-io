CC=gcc
PGCC=hipcc
CFLAGS=-lm
PGCCFLAGS=-O3 -x hip -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 -D__HIP_PLATFORM_AMD__ --rocm-path=${ROCM_PATH} --offload-arch=gfx90a

LIBS =

# hipCOMP-core configuration
ifdef USE_HIPCOMP
    HIPCOMP_PATH = $(HIPCOMP_ROOT)
    CFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    PGCCFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    LIBS += -L$(HIPCOMP_PATH)/lib -lhipcomp -Wl,-rpath,$(HIPCOMP_PATH)/lib
endif

LIBS += -L${ROCM_PATH}/lib -lamdhip64 -lstdc++ $(GCC_LIBS)
