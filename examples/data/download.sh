#!/bin/bash
wget http://www.damp.tottori-u.ac.jp/~hoshi/elses_matrix/ELSES_MATRIX_BNZ30_20180227.tgz
wget http://www.damp.tottori-u.ac.jp/~hoshi/elses_matrix/ELSES_MATRIX_DIAB18h_20130430.tgz
wget http://www.damp.tottori-u.ac.jp/~hoshi/elses_matrix/ELSES_MATRIX_VCNT400std_20130515.tgz

tar -xzf ELSES_MATRIX_BNZ30_20180227.tgz --strip-components=1 -C . "ELSES_MATRIX_BNZ30_20180227/ELSES_MATRIX_BNZ30_A.mtx" "ELSES_MATRIX_BNZ30_20180227/ELSES_MATRIX_BNZ30_B.mtx" && rm -f ELSES_MATRIX_BNZ30_20180227.tgz
tar -xzf ELSES_MATRIX_DIAB18h_20130430.tgz --strip-components=1 -C . "ELSES_MATRIX_DIAB18h_20130430/ELSES_MATRIX_DIAB18h_A.mtx" "ELSES_MATRIX_DIAB18h_20130430/ELSES_MATRIX_DIAB18h_B.mtx" && rm -f ELSES_MATRIX_DIAB18h_20130430.tgz
tar -xzf ELSES_MATRIX_VCNT400std_20130515.tgz --strip-components=1 -C . "ELSES_MATRIX_VCNT400std_20130515/ELSES_MATRIX_VCNT400std_A.mtx" && rm -f ELSES_MATRIX_VCNT400std_20130515.tgz
