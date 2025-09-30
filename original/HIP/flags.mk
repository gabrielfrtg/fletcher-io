CC=$(GCC)
CFLAGS=-O3
GPUCC=$(HIPCC)
GPUCFLAGS=$(HIPCFLAGS)

LIBS =

# hipCOMP configuration
ifdef USE_HIPCOMP
    HIPCOMP_PATH ?= $(HIPCOMP_ROOT)
    CFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP
    GPUCFLAGS += -I$(HIPCOMP_PATH)/include -DUSE_HIPCOMP

    HIPCOMP_LIB_DIR ?=
    ifndef HIPCOMP_LIB_DIR
        HIPCOMP_LIB_CANDIDATES := \
            lib \
            lib64 \
            build/lib \
            build/lib64 \
            install/lib \
            install/lib64
        HIPCOMP_LIB_DIR := $(firstword \
            $(foreach dir,$(HIPCOMP_LIB_CANDIDATES),\
                $(if $(wildcard $(HIPCOMP_PATH)/$(dir)/libhipcomp.so),$(HIPCOMP_PATH)/$(dir),)) \
            $(foreach dir,$(HIPCOMP_LIB_CANDIDATES),\
                $(if $(wildcard $(HIPCOMP_PATH)/$(dir)/libhipcomp.a),$(HIPCOMP_PATH)/$(dir),)))
    endif
    ifeq ($(strip $(HIPCOMP_LIB_DIR)),)
        HIPCOMP_LIB_DIR := $(HIPCOMP_PATH)/lib
    endif

    LIBS += -L$(HIPCOMP_LIB_DIR) -lhipcomp -Wl,-rpath,$(HIPCOMP_LIB_DIR)
endif

LIBS += $(HIPCC_LIBS) $(GCC_LIBS)
