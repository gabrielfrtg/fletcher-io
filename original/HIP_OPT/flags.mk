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

# Compression algorithm selection: LZ4 (default) or ZFP
# Usage: make backend=HIP_OPT USE_HIPCOMP=1 COMP_ALGO=ZFP
#
# ZFP GPU LIMITATION: Only FIXED_RATE (lossy) mode is supported on GPU!
# Reversible/lossless mode is NOT supported by ZFP CUDA/HIP backend.
# See: https://zfp.readthedocs.io/en/release1.0.1/execution.html#cuda-limitations
#
# ZFP_RATE controls compression (bits per value):
#   32.0 = minimal loss (~1x compression)
#   24.0 = very low loss (~1.33x compression) [DEFAULT - error ~1e-5]
#   16.0 = low loss (~2x compression, error ~1e-3)
#   8.0  = moderate loss (~4x compression)
#   4.0  = higher loss (~8x compression)
COMP_ALGO ?= LZ4
ZFP_RATE ?= 24.0

# hipCOMP-core configuration
ifdef USE_HIPCOMP
    HIPCOMP_PATH = $(HIPCOMP_ROOT)
    CFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    PGCCFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    LIBS += -L$(HIPCOMP_PATH)/lib -lhipcomp -Wl,-rpath,$(HIPCOMP_PATH)/lib
    
    # Define compression algorithm
    ifeq ($(COMP_ALGO),ZFP)
        CFLAGS += -DCOMP_ALGO_ZFP -DZFP_RATE=$(ZFP_RATE)
        PGCCFLAGS += -DCOMP_ALGO_ZFP -DZFP_RATE=$(ZFP_RATE)
        # ZFP requires additional library
        LIBS += -lzfp_hip_impl
    else
        CFLAGS += -DCOMP_ALGO_LZ4
        PGCCFLAGS += -DCOMP_ALGO_LZ4
    endif
endif

LIBS += -L${ROCM_PATH}/lib -lamdhip64 -lstdc++ $(GCC_LIBS)
