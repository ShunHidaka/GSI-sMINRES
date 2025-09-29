/**
 * \file blas_zhpmv.hpp
 * \brief BLAS wrappers for the GSI-MINRES sample program.
 * \author Shuntaro Hidaka
 *
 * \details 
 */

#ifndef GSI_SMINRES_LINALG_BLAS_ZHPMV_HPP
#define GSI_SMINRES_LINALG_BLAS_ZHPMV_HPP

#include "gsi_sminres/linalg/blas.hpp"
#include <cblas.h>

namespace gsi_sminres {
  namespace linalg {
    namespace blas {
      inline CBLAS_UPLO to_cblas_uplo(char ul) noexcept {
        return (ul == 'U' || ul == 'u') ? CblasUpper : CblasLower;
      }
      // ========== Level-2: mv ==========
      /**
       * \brief Hermitian packed 'U' matrix-vector multiplication: \f$ y = \alpha A x + \beta y \f$.
       * \param[in]     uplo     Which triangle of \p A is stored: 'U' (upper) or 'L' (lower).
       * \param[in]     n        Problem dimension (length of logical vectors \p x and \p y).
       * \param[in]     alpha    Scalar multiplier for A*x.
       * \param[in]     ap       Packed Hermitian storage of \p A, size \f$ n(n+1)/2 \f$.
          *                      The data are assumed contiguous and start at \c ap.data().
       * \param[in]     x        Input vector buffer (containing \p x).
       * \param[in]     x_offset Element offset to the first logical entry of \p x (0-based).
       * \param[in]     beta     Scalar multiplier for y.
       * \param[in,out] y        In/out vector buffer (overwritten with the result).
       * \param[in]     y_offset Element offset to the first logical entry of \p y (0-based).
       * \param[in]     incx     Stride (in elements) between consecutive entries of \p x (default 1).
       * \param[in]     incy     Stride (in elements) between consecutive entries of \p y (default 1).
       */
      inline void zhpmv(char uplo,std::size_t n,
                        std::complex<double> alpha,
                        const std::vector<std::complex<double>>& ap,
                        const std::vector<std::complex<double>>& x, std::size_t x_offset,
                        std::complex<double> beta,
                        std::vector<std::complex<double>>&       y, std::size_t y_offset,
                        std::size_t incx=1, std::size_t incy=1) noexcept {
        const blas_int   nn = to_blas_int(n);
        const blas_int   ix = to_blas_int(incx);
        const blas_int   iy = to_blas_int(incy);
        cblas_zhpmv(CblasColMajor, to_cblas_uplo(uplo), nn,
                    &alpha, ap.data(), x.data()+x_offset, ix,
                    &beta, y.data()+y_offset, iy);
      }


    }  // namespace blas
  }  // namespace linalg
}  // namespace gsi_sminres

#endif // GSI_SMINRES_LINALG_BLAS_ZHPMV_HPP
