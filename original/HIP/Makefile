include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c cuda_driver.c
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c cuda_stuff.cu
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c cuda_propagate.cu
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c cuda_insertsource.cu
ifdef USE_HIPCOMP
	$(GPUCC) $(GPUCFLAGS) $(COMMON_FLAGS) -c cuda_compression.cu
endif

clean:
	rm -f *.o *.a

