/**
 * \file mm_zhp.hpp
 * \brief I/O helpers: Matrix Market (symmetric/Hermitian) -> BLAS packed (ZHP) for GSI-sMINRES samples
 * \author Shuntaro Hidaka
 *
 * \details
 *   Declares a loader that converts a Matrix Market file describing a symmetric/Hermitian
 *   square matrix into the BLAS/LAPACK complex Hermitian packed storage (ZHP).
 *   The packed layout is chosen as either upper ('U') or lower ('L').
 *
 * \notes CblasColMajor
 */

#ifndef GSI_SMINRES_EXTRAS_IO_MM_ZHP_HPP
#define GSI_SMINRES_EXTRAS_IO_MM_ZHP_HPP

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace gsi_sminres {
  namespace io {

    /**
     * \brief Load a Hermitian matrix from a Matrix Market file into BLAS packed (ZHP) storage.
     * \param[in]  filename Path to the Matrix Market file.
     * \param[out] size     Matrix order n (square nxn).
     * \param[out] uplo     Which triangular part is stored in the packed array:
     *                      set to 'U' (upper) or 'L' (lower).
     *
     * \return A std::vector<std::complex<double>> of length n(n+1)/2 holding the
     *         packed Hermitian matrix A in ZHP format.
     *
     * \note 
     *   - Input may be real-symmetric or complex-Hermitian in Matrix Market (coordinate) form.
     *   - The result uses the BLAS/LAPACK "packed" convention:
     *       * If uplo == 'U': element (i,j) with 0 ≤ i ≤ j < n is stored at
     *         index k = j*(j+1)/2 + i.
     *       * If uplo == 'L': element (i,j) with 0 ≤ j ≤ i < n is stored at
     *         index k = i*(i+1)/2 + j.
     *   - Real-symmetric inputs are represented as complex with zero imaginary parts.
     *
     * \note Error handling policy:
     *        This function may terminate the program on severe I/O/format errors in the
     *        sample implementation. In production code, consider throwing exceptions instead.
     */
    std::vector<std::complex<double>> load_mm_zhp(const std::string& filename,
                                                  std::size_t&       size,
                                                  char&              uplo);

  }  // namespace io
}  // namespace gsi_minres

#endif // GSI_SMINRES_EXTRAS_IO_MM_ZHP_HPP
