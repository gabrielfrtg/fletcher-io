include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmp_driver.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmp_propagate.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openmp_insertsource.c

clean:
	rm -f *.o *.a
