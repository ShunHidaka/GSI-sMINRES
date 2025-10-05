# GSI-sMINRES

A high-performance C++17 solver suite for multi-shift Hermitian linear systems (MINRES-based).

## Overview

GSI-sMINRES is a **shifted MINRES** solver suite implemented in C++17.  
It efficiently solves **multiple shifts simultaneously** for matrices $A$ and $B$ that are **real-symmetric** or **complex-Hermitian**. The project prioritizes performance, low memory footprint, and portability, exposing a simple C++ API backed by BLAS.

**Standard shifted linear systems:**

$$ (A + \sigma^{(m)} I)\,\mathbf{x}^{(m)} = \mathbf{b}, \quad (m=1,2,\dots,M) $$

**Generalized shifted linear systems:**

$$ (A + \sigma^{(m)} B)\,\mathbf{x}^{(m)} = \mathbf{b}, \quad (m=1,2,\dots,M) $$

Provided solvers (three families):
- **Standard shifted MINRES**
- **Generalized shifted MINRES**
- **Shift-Invert preconditioned shifted MINRES**

> For algorithmic background, derivations, and comparisons among methods, see the “Algorithm” chapter in the Doxygen documentation.

## Documents

- Manual ([HTML](https://shunhidaka.github.io/GSI-sMINRES/)/PDF)

---

## Requirements

- C++17 compiler (GCC ≥ 9 / Clang ≥ 10 recommended)
- **BLAS** (Netlib / OpenBLAS / MKL, etc.) — required by the core
- **LAPACK** — used by examples / extras
- CMake ≥ 3.18 and `make`
- (Optional) OpenMP (depending on your BLAS and whether you parallelize SpMV)

---

## Directory Structure (excerpt)

```
GSI-sMINRES/
├── CMakeLists.txt
├── include/gsi_sminres/
│   ├── algorithms/        # standard / generalized / shift_invert
│   ├── extras/            # io (MatrixMarket), sparse (CSR/SpMV), small algos
│   └── linalg/            # BLAS/LAPACK wrappers
├── src/                   # implementations
└── examples/              # example_{std,gen,sis}_{zhp,csr}.cpp + data/
```

---

## Installation (CMake recommended)

```bash
# Configure & build
cmake -S . -B build
cmake --build build -j

# Install (default prefix e.g. $HOME/gsminres_install)
cmake --install build
```

### Common CMake Options

| Option                               | Default                      | Description                  |
|--------------------------------------|------------------------------|------------------------------|
| `-DCMAKE_INSTALL_PREFIX=...`         | `"$HOME/gsminres_install"`   | Install destination          |
| `-DGSI_SMINRES_BUILD_EXAMPLES=ON`    | `ON`                         | Build example programs       |
| `-DUSE_OPENMP`                       | `ON`                         | Enable OpenMP if available.  |

> If BLAS/LAPACK are not auto-detected, consider adding `CMAKE_PREFIX_PATH` or explicitly setting `BLAS_LIBRARIES` / `LAPACK_LIBRARIES`.

---

## Quick Start

### 1) Manual linking (g++)

Assuming it is installed under `$HOME/gsminres_install`:
```bash
g++ -std=gnu++17 -O3 -march=native \
  -I"$HOME/gsminres_install/include" \
  myprog.cpp \
  -L"$HOME/gsminres_install/lib" -lgsisminres \
  -lblas -llapack \
  -Wl,-rpath,"$HOME/gsminres_install/lib"
```

### 2) Use CMake (`find_package`)

**`CMakeLists.txt` in your project**:
```cmake
cmake_minimum_required(VERSION 3.18)
project(myapp CXX)

find_package(gsi_sminres CONFIG REQUIRED)

add_executable(myapp src/myapp.cpp)
target_link_libraries(myapp PRIVATE gsi_sminres::gsi_sminres)
```

Build example:
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/gsminres_install"
cmake --build build -j
```

---

## Running the Examples

This repository provides two example pathways: **ZHP (packed Hermitian, BLAS/LAPACK)** and **CSR (sparse, in-house SpMV/CG)**.
You can fetch sample matrix data via `examples/data/download.sh`.

```bash
# Example: standard shift with the packed Hermitian (BLAS/LAPACK) pathway
./build/example_std_zhp  examples/data/A.mtx

# Example: generalized shift with the CSR (sparse) pathway
./build/example_gen_csr  examples/data/A.mtx  examples/data/B.mtx

# Example: shift-invert preconditioned
./build/example_sis_zhp  examples/data/A.mtx  examples/data/B.mtx
```

---

## API Entry Points (overview)

- Public headers: `include/gsi_sminres/...`  
- Typical workflow:
  1. Construct the solver (matrix size, number of shifts, etc.)
  2. Initialize (initial guess, shift set, tolerances)
  3. Provide **matrix–vector products** and **inner solves** as required by the chosen pathway
  4. Iterate until convergence → obtain solution vectors and residual metrics

> For exact function signatures and advanced controls (stopping criteria, multi-shift updates, etc.), please refer to the Doxygen manual.

---

## Known Notes

- **OpenBLAS `zrotg`**: versions < 0.3.27 have a known issue affecting complex Givens rotations.  
  - See: https://github.com/OpenMathLib/OpenBLAS/issues/4909
  - **Workarounds**
    - Update OpenBLAS version 0.3.27 or later.
    - Use an alternative BLAS implementation (e.g., Netlib BLAS or Intel MKL).

---

## License

MIT License (see `LICENSE`).

---

## Acknowledgments & Citation

If you use this code, please cite:

```bibtex
@misc{gsi_sminres,
  author       = {},
  title        = {},
  howpublished = {},
  year         = {},
  note         = {}
}
```
