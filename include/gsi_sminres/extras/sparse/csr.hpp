/**
 * \file csr.hpp
 * \brief Minimal CSR (Compressed Sparse Row) struct used by GSI-sMINRES sample programs.
 * \author Shuntaro Hidaka
 *
 * \details 
 *   Plain data holder for a square sparse matrix in CSR format. Indices are 0-based.
 *   The structure does not enforce invariants at runtime;
 *   callers are responsible for providing consistent arrays.
 *
 *   Expected invariants (for correctness and performance):
 *     - row_ptr.size() == n + 1
 *     - col_idx.size() == values.size()  (== nnz)
 *     - row_ptr is non-decreasing and row_ptr[0] == 0, row_ptr[n] == nnz
 *     - 0 <= col_idx[k] < n  for all 0 <= k < nnz
 *     - (Recommended) column indices in each row are strictly increasing
 *       to enable fast searches/merges.
 *
 *   Notes:
 *     - This struct does not imply any symmetry/hermiticity.
 *     - No memory ownership beyond std::vector; no methods are provided on purpose.
 */

#ifndef GSI_SMINRES_EXTRAS_SPARSE_CSR_HPP
#define GSI_SMINRES_EXTRAS_SPARSE_CSR_HPP

#include <complex>
#include <cstddef>
#include <vector>

namespace gsi_sminres {
  /**
   * \namespace gsi_sminres::sparse
   * \brief [Utility] Sparse utilities for GSI-sMINRES.
   * \details Currently includes CSRMatrix and a CSR-based SpMV routine (y <- A x).
   *          Designed to remain header-only and back-end agnostic.
   */
  namespace sparse {

    /**
     * \brief Complex-valued CSR sparse matrix (square).
     * \details nnz := Number of NonZero elements.
     */
    struct CSRMatrix {
      std::size_t n;                            ///< matrix dimension (n x n, 0-based).
      std::vector<std::size_t> row_ptr;         ///< row pointer array (size = n+1, 0-based).
      std::vector<std::size_t> col_idx;         ///< column index array (size = nnz).
      std::vector<std::complex<double>> values; ///< Nonzero values array (size = nnz, aligned with col_idx).
    };

    /**
     * \brief Real-valued CSR sparse matrix (square).
     * \details nnz := Number of NonZero elements.
     */
    struct CSRMatrix_r {
      std::size_t n;                    ///< matrix dimension (n x n, 0-based).
      std::vector<std::size_t> row_ptr; ///< row pointer array (size = n+1, 0-based).
      std::vector<std::size_t> col_idx; ///< column index array (size = nnz).
      std::vector<double> values;       ///< Nonzero values array (size = nnz, aligned with col_idx).
    };

  }  // namespace sparse
}  // namespace gsi_sminres

#endif // GSI_SMINRES_EXTRAS_SPARSE_CSR_HPP
