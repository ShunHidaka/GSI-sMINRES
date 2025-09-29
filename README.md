# GSI-sMINRES

---

## Overview


---

## Requirement

- C++17 compiler
- BLAS library
- LAPACK library (required for sample programs, but **complication will fail without it in some BLAS/LAPACK**)
- `make`
- `CMake` (Storongly recommend)

---

## Documents

- Manual ([HTML](https://shunhidaka.github.io/GSMINRESpp/)/PDF)

---

## Directory Structure

```
.
├── CMakeLists.txt
├── Doxyfile
├── README.md
├── build
├── cmake
│   ├── gsi_sminresConfig.cmake.in
├── examples
│   ├── Makefile
│   ├── data
│   │   ├── download.sh
│   ├── example_gen_csr.cpp
│   ├── example_gen_zhp.cpp
│   ├── example_sis_csr.cpp
│   ├── example_sis_zhp.cpp
│   ├── example_std_csr.cpp
│   ├── example_std_zhp.cpp
├── include
│   ├── gsi_sminres
│   │   ├── algorithms
│   │   │   ├── generalized_shifted_minres.hpp
│   │   │   ├── shift_invert_shifted_minres.hpp
│   │   │   ├── standard_shifted_minres.hpp
│   │   ├── extras
│   │   │   ├── algorithms
│   │   │   │   ├── cg.hpp
│   │   │   │   ├── minres_pencil.hpp
│   │   │   ├── io
│   │   │   │   ├── mm_csr.hpp
│   │   │   │   ├── mm_zhp.hpp
│   │   │   ├── sparse
│   │   │   │   ├── csr.hpp
│   │   │   │   ├── spmv.hpp
│   │   ├── gsi_sminres.hpp
│   │   ├── linalg
│   │   │   ├── blas.hpp
│   │   │   ├── blas_zhpmv.hpp
│   │   │   ├── lapack.hpp
├── src
│   ├── algorithms
│   │   ├── generalized_shifted_minres.cpp
│   │   ├── shift_invert_shifted_minres.cpp
│   │   ├── standard_shifted_minres.cpp
│   ├── extras
│   │   ├── algorithms
│   │   │   ├── cg.cpp
│   │   │   ├── minres_pencil.cpp
│   │   ├── io
│   │   │   ├── mm_csr.cpp
│   │   │   ├── mm_zhp.cpp
```

---

## Installation

### Using CMake (recommended)
``` bash
mkdir build
cmake -S . -B build
cd build/
make           # Build sample programs and libraries
make install   # Install to $HOME/gsminres_install by default
```
See the [manual](hogehoge) for detailed instructions.

---

## How to link this library

### Shared library
hogehoge
``` bash
# Standard Complication
$ g++ myprog.cpp -
```

### Static library
ホームディレクトリに `gsi-sminres` がインストール済みであるとする
``` bash
# Standard Complication
$ g++ -std=c++17 -O3 -I
```

---

## API Summary
See the [manual]().

---

## Known Issues
- OpenBLAS versions prior to 0.3.27 has bug in the `zrotg`.
  - See: https://github.com/OpenMathLib/OpenBLAS/issues/4909
  - **Workarounds**
    - Update OpenBLAS version 0.3.27 or later.
    - Use an alternative BLAS implementation (e.g., Netlib BLAS or Interl MKL).
    - Optionally, modify the source ~~~

---

## Citation
If you use this code, please cite:
``` bibtex
@article{
  author  = {},
  title   = {},
  doi     = {},
  journal = {},
  volume  = {},
  pages   = {},
  year    = {}
}
```

---

## Licnse
MIT License