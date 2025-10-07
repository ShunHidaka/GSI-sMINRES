CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
ROOT     ?= ..
INC      := -I$(ROOT)/include
BLAS_LIB := -lblas -llapack

USE_OPENMP ?= 1
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

ALG_STD := $(ROOT)/src/algorithms/standard_shifted_minres.cpp
ALG_GEN := $(ROOT)/src/algorithms/generalized_shifted_minres.cpp
ALG_SIS := $(ROOT)/src/algorithms/shift_invert_shifted_minres.cpp
IO_CSR  := $(ROOT)/src/extras/io/mm_csr.cpp
IO_ZHP  := $(ROOT)/src/extras/io/mm_zhp.cpp
ALG_CG  := $(ROOT)/src/extras/algorithms/cg.cpp
ALG_MR  := $(ROOT)/src/extras/algorithms/minres_pencil.cpp

.PHONY: all std gen sis clean
all: std gen sis

std: a_std_csr.out a_std_zhp.out
a_std_csr.out: example_std_csr.cpp $(ALG_STD) $(IO_CSR)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(BLAS_LIB)
a_std_zhp.out: example_std_zhp.cpp $(ALG_STD) $(IO_ZHP)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(BLAS_LIB)

gen: a_gen_csr.out a_gen_zhp.out
a_gen_csr.out: example_gen_csr.cpp $(ALG_GEN) $(IO_CSR) $(ALG_CG)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(BLAS_LIB)
a_gen_zhp.out: example_gen_zhp.cpp $(ALG_GEN) $(IO_ZHP)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(BLAS_LIB)

sis: a_sis_csr.out a_sis_zhp.out
a_sis_csr.out: example_sis_csr.cpp $(ALG_SIS) $(IO_CSR) $(ALG_MR)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(BLAS_LIB)
a_sis_zhp.out: example_sis_zhp.cpp $(ALG_SIS) $(IO_ZHP)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(BLAS_LIB)

clean:
	rm -f a_*.out
