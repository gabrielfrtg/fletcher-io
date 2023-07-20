CC=$(PGCC)
CFLAGS=-O3 -acc -ta=tesla:$(PGCC_GPU_SM),multicore
LIBS=$(PGCC_LIBS)