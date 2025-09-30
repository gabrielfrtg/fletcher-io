# config include file for fletcher
# common for all backends

# Compilers
GCC=gcc
NVCC=nvcc
HIPCC=hipcc
# PGCC=pgcc
# CLANG=clang

# Library paths
GCC_LIBS=-lm
NVCC_LIBS=-lcudart -lstdc++    # it may include CUDA lib64 path...

# Detect the ROCm installation to ensure we can find libamdhip64 when linking.
ROCM_PATH ?= $(shell hipconfig --rocmpath 2>/dev/null)
ifeq ($(strip $(ROCM_PATH)),)
ROCM_PATH ?= /opt/rocm
endif

HIPCC_LIBS=-L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lstdc++
HIPCFLAGS=-O3
PGCC_LIBS=-lm
# CLANG_LIBS=-lm

# PAPI flags
PAPI_LIBS=-lpapi
