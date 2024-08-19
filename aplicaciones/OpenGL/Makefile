CUDA_PATH=/usr/local/cuda-11.0
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=simpleGL

NVCC_FLAGS=-m64 # --resource-usage

LIBRARIES = -lglut -lGL -lGLU

GENCODE_FLAGS = -arch=sm_50

all : ${TARGET}

simpleGL: simpleGL.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o
