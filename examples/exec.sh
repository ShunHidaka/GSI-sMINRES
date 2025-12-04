#!/bin/bash
set -x

# Standard
./a_std_csr.out   data/ELSES_MATRIX_VCNT400std_A.mtx
./a_std_csr_r.out data/ELSES_MATRIX_VCNT400std_A.mtx
./a_std_zhp.out   data/ELSES_MATRIX_VCNT400std_A.mtx

# Generalized
## Real-symmetric
./a_gen_csr.out   data/ELSES_MATRIX_BNZ30_A.mtx data/ELSES_MATRIX_BNZ30_B.mtx
./a_gen_csr_r.out data/ELSES_MATRIX_BNZ30_A.mtx data/ELSES_MATRIX_BNZ30_B.mtx
./a_gen_zhp.out   data/ELSES_MATRIX_BNZ30_A.mtx data/ELSES_MATRIX_BNZ30_B.mtx
# Hermitian
./a_gen_csr.out   data/ELSES_MATRIX_DIAB18h_A.mtx data/ELSES_MATRIX_DIAB18h_B.mtx
./a_gen_csr_r.out data/ELSES_MATRIX_DIAB18h_A.mtx data/ELSES_MATRIX_DIAB18h_B.mtx
./a_gen_zhp.out   data/ELSES_MATRIX_DIAB18h_A.mtx data/ELSES_MATRIX_DIAB18h_B.mtx

# Shift-invert preconditioned
## Real-symmetric
./a_sis_csr.out   data/ELSES_MATRIX_BNZ30_A.mtx data/ELSES_MATRIX_BNZ30_B.mtx
./a_sis_csr_r.out data/ELSES_MATRIX_BNZ30_A.mtx data/ELSES_MATRIX_BNZ30_B.mtx
./a_sis_zhp.out   data/ELSES_MATRIX_BNZ30_A.mtx data/ELSES_MATRIX_BNZ30_B.mtx
# Hermitian
./a_sis_csr.out   data/ELSES_MATRIX_DIAB18h_A.mtx data/ELSES_MATRIX_DIAB18h_B.mtx
./a_sis_csr_r.out data/ELSES_MATRIX_DIAB18h_A.mtx data/ELSES_MATRIX_DIAB18h_B.mtx
./a_sis_zhp.out   data/ELSES_MATRIX_DIAB18h_A.mtx data/ELSES_MATRIX_DIAB18h_B.mtx
