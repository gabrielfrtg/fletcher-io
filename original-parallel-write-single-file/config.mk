# config include file for fletcher
# common for all backends

# Compilers
GCC=gcc
NVCC=nvcc
# PGCC=pgcc
# CLANG=clang

# Library paths
GCC_LIBS=-lm
NVCC_LIBS=-lcudart -lstdc++    # it may include CUDA lib64 path...
PGCC_LIBS=-lm
# CLANG_LIBS=-lm

# PAPI flags
PAPI_LIBS=-lpapi
