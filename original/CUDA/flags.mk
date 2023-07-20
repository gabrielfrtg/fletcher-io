CC=$(GCC)
CFLAGS=-O3
GPUCC=$(NVCC)
GPUCFLAGS=-Xptxas="-v" --maxrregcount 127 --gpu-architecture sm_75 # $(CUDA_GPU_SM)
LIBS=-L/usr/local/cuda/lib64 -lcudart -lstdc++ $(GCC_LIBS)
