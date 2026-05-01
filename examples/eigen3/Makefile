CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O3 -DNDEBUG
ROOT      ?= ../..
INC       := -I$(ROOT)/include
INC_EIGEN := -I/usr/include/eigen3
BLAS_LIB  := -lblas -llapack

USE_OPENMP ?= 1
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

ALG_GEN := $(ROOT)/src/algorithms/generalized_shifted_minres.cpp
ALG_SIS := $(ROOT)/src/algorithms/shift_invert_shifted_minres.cpp
IO_CSR  := $(ROOT)/src/extras/io/mm_csr.cpp

.PHONY: all eigen clean
all: eigen

eigen: a_gen_eigen.out a_sis_eigen.out
a_gen_eigen.out: example_gen_eigen.cpp $(ALG_GEN) $(IO_CSR)
	$(CXX) $(CXXFLAGS) $(INC) $(INC_EIGEN) $^ -o $@ $(BLAS_LIB)
a_sis_eigen.out: example_sis_eigen.cpp $(ALG_SIS) $(IO_CSR)
	$(CXX) $(CXXFLAGS) $(INC) $(INC_EIGEN) $^ -o $@ $(BLAS_LIB) 

clean:
	rm -f a_*.out
