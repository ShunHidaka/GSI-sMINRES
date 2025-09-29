/**
 * \file mm_csr.hpp
 * \brief I/O helpers: Matrix Market (symmetric/Hermitian) -> Compressed Sparse Row (CSR) for GSI-sMINRES samples
 * \author Shuntaro Hidaka
 *
 * \details
 *   Loads a real-symmetric or complex-Hermitian square matrix given in Matrix Market
 *   coordinate format and returns a struct 'CSRMatrix'.
 *   Indices are converted to 0-based, and duplicate entries are accumulated.
 *   On any error this function prints a message and terminates the process.
 */

#ifndef GSI_SMINRES_EXTRAS_IO_MM_CSR_HPP
#define GSI_SMINRES_EXTRAS_IO_MM_CSR_HPP

#include <cstddef>
#include <complex>
#include <string>
#include <vector>
#include "gsi_sminres/extras/sparse/csr.hpp"

namespace gsi_sminres {
  namespace io {

    /**
     * \brief Load Matrix Market (real-symmetric or complex-Hermitian, coordinate) and return a full CSR matrix.
     * \param[in]  filename  Path to the Matrix Market (.mtx) file.
     * \param[out] n         Matrix order (n x n).
     * \return CSRMatrix.
     *
     * \note
     *   - Only real symmetric or complex Hermitian kinds are supported.
     *   - Duplicate entries are summed.
     *   - Real-symmetric inputs are represented as complex with zero imaginary parts.
     *
     * \note Error handling policy:
     *        This function may terminate the program on severe I/O/format errors in the
     *        sample implementation. In production code, consider throwing exceptions instead.
     */
    gsi_sminres::sparse::CSRMatrix load_mm_csr(const std::string& filename,
                                               std::size_t&       size);

  }  // namespace io
}  // namespace gsi_minres

#endif // GSI_SMINRES_EXTRAS_IO_MM_CSR_HPP
