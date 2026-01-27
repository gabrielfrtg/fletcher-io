include ../config.mk
include flags.mk

all:	
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c hip_driver.c
	$(PGCC) $(PGCCFLAGS) $(COMMON_FLAGS) -c hip_stuff.c
	$(PGCC) $(PGCCFLAGS) $(COMMON_FLAGS) -c hip_propagate.c
	$(PGCC) $(PGCCFLAGS) $(COMMON_FLAGS) -c hip_insertsource.c
ifdef USE_HIPCOMP
	$(PGCC) $(PGCCFLAGS) $(COMMON_FLAGS) -c hip_compression.c
endif

clean:
	rm -f *.o *.a
