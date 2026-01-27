CC=gcc
PGCC=hipcc
CFLAGS=-lm -fPIC
# GPU Architecture: gfx1100 = Radeon RX 7900 XT/XTX (RDNA 3)
# Other options:
#   gfx942  = MI300X (CDNA 3)
#   gfx90a  = MI250/MI250X (CDNA 2)
#   gfx908  = MI100 (CDNA 1)
#   gfx906  = MI50/MI60
GPU_ARCH ?= gfx1100
PGCCFLAGS=-O3 -fPIC -x hip -D__HIP_ROCclr__ -D__HIP_ARCH_GFX1100__=1 -D__HIP_PLATFORM_AMD__ --rocm-path=${ROCM_PATH} --offload-arch=$(GPU_ARCH)

LIBS =

# hipCOMP-core configuration
ifdef USE_HIPCOMP
    HIPCOMP_PATH = $(HIPCOMP_ROOT)
    CFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    PGCCFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    LIBS += -L$(HIPCOMP_PATH)/lib -lhipcomp -Wl,-rpath,$(HIPCOMP_PATH)/lib
endif

LIBS += -L${ROCM_PATH}/lib -lamdhip64 -lstdc++ $(GCC_LIBS)
