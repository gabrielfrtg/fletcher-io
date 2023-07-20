# Environment and compilation flags at Exaflop
#   module load gompi

# Default
backend=CUDA
arch=$(backend)

include config.mk
include $(arch)/flags.mk

OBJ1=\
	source.o \
	utils.o \
	boundary.o \
	walltime.o \
	model.o \
	map.o

ifdef PAPI
	LIBS += $(PAPI_LIBS)
	OBJ1 += ModPAPI.o
	CFLAGS += -DPAPI
endif

OBJ=main.o $(OBJ1)

TARGET = ModelagemFletcher.exe

all: 	$(OBJ)
	cd $(arch) && make
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(arch)/*.o $(LIBS)

main.o:	main.c $(OBJ1)
	$(CC) -c $(CFLAGS) main.c

boundary.o:	boundary.c boundary.h  map.o
	$(CC) -c $(CFLAGS) boundary.c

source.o:	source.c source.h
	$(CC) -c $(CFLAGS) source.c

utils.o:	utils.c utils.h map.o source.o
	$(CC) -c $(CFLAGS) utils.c

map.o:	map.c map.h
	$(CC) -c $(CFLAGS) map.c

model.o:	model.c model.h
	$(CC) -c $(CFLAGS) $(COMMON_FLAGS) model.c

walltime.o:	walltime.c walltime.h
	$(CC) -c $(CFLAGS) walltime.c

ModPAPI.o:	ModPAPI.c ModPAPI.h
	$(CC) -c $(CFLAGS) ModPAPI.c


compare.exe:	compare.c
	gcc compare.c -o compare.exe

.SUFFIXES	:	.o .c

.c.o:
	$(CC) $(CFLAGS) $*.c

clean:
	cd $(arch) && make clean
	rm -f *.o $(TARGET)

clean-all:
	rm -f */*.o *.o $(TARGET)
