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
HIPCC_LIBS=-lamdhip64 -lstdc++
HIPCFLAGS=-O3
PGCC_LIBS=-lm
# CLANG_LIBS=-lm

# PAPI flags
PAPI_LIBS=-lpapi
