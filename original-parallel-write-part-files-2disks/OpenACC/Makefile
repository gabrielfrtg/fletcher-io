include ../config.mk
include flags.mk

all:
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openacc_driver.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openacc_propagate.c
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c openacc_insertsource.c

clean:
	rm -f *.o *.a
