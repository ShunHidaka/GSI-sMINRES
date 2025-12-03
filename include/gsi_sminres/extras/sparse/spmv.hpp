/**
 * \file spmv.hpp
 * \brief Sparse matrix–vector multiplication (SpMV): y <- A x for CSRMatrix.
 *
 * \details
 *   Minimal SpMV used by GSI-sMINRES samples.
 *   - Input:
 *       A : CSRMatrix (square, 0-based indices)
 *       x : input vector (length == A.n)
 *   - Output:
 *       y : result vector (length == A.n), overwritten with A*x
 *   - Parallelism:
 *       Row-wise OpenMP parallelization; each iteration writes to distinct y[i].
 *   - Preconditions (not checked here for overhead minimization):
 *       * A.row_ptr.size() == A.n + 1
 *       * A.col_idx.size() == A.values.size() (== nnz)
 *       * x.size() == A.n, y.size() == A.n
 *       * 0 <= A.col_idx[k] < A.n for all nonzeros
 *   - Notes:
 *       This kernel assumes a general CSR (no symmetry expansion).
 */

#ifndef GSI_SMINRES_EXTRAS_SPARSE_SPMV_HPP
#define GSI_SMINRES_EXTRAS_SPARSE_SPMV_HPP

#include <vector>
#include <complex>
#include <cstddef>
#if defined(_OPENMPE)
#include <omp.h>
#endif

#include "gsi_sminres/extras/sparse/csr.hpp"

namespace gsi_sminres {
  namespace sparse {

    /**
     * \brief Compute y = A x (Composed Sparse Row, complex-value).
     * \param[in]  A CSRMatrix (square)
     * \param[in]  x input vector (size = A.n)
     * \param[out] y result vector (size = A.n)
     */
    inline void SpMV(const CSRMatrix& A,
                     const std::vector<std::complex<double>>& x,
                     std::vector<std::complex<double>>& y) {
      std::size_t n = A.n;
#pragma omp parallel for schedule(static)
      for (std::size_t i = 0; i < n; ++i) {
        std::complex<double> sum = {0.0, 0.0};
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
          sum += A.values[j] * x[A.col_idx[j]];
        }
        y[i] = sum;
      }
    }

    /**
     * \brief Compute y = A x (Composed Sparse Row, real-value).
     * \param[in]  A CSRMatrix (square)
     * \param[in]  x input vector (size = A.n)
     * \param[out] y result vector (size = A.n)
     */
    inline void SpMV_r(const CSRMatrix_r& A,
                     const std::vector<double>& x,
                     std::vector<double>& y) {
      std::size_t n = A.n;
#pragma omp parallel for schedule(static)
      for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
          sum += A.values[j] * x[A.col_idx[j]];
        }
        y[i] = sum;
      }
    }

  }  // namespace sparse
}  // namespace gsi_sminres

#endif // GSI_SMINRES_EXTRAS_SPARSE_SPMV_HPP
