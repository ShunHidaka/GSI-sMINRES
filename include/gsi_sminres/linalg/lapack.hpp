/**
 * \file lapack.hpp
 * \brief LAPACK wrapper functions for the GSI-sMINRES sample programs.
 * \author Shuntaro Hidaka
 *
 * \details This header provides minimal C++ wrappers for selected LAPACK routines:
 *           - ZPPTRF : Cholesky factorization of Hermitian positive definited packed A
 *           - ZPPTRS : Solve Ax=b using the ZPPTRF factorization
 *           - ZHPTRF : Bunch-Kaufman factorization of Hermitian packed A
 *           - ZHPTRS : Solve Ax=b using the ZHPTRF factorization
 *           - ZLARTG : Robust Givens rotation parameters (c,s,r) for complex f,g
 *
 *          Notes:
 *           - Column-major semantics (Fortran LAPACK).
 *           - No size/value checks and no exceptions; \c info is returned to the caller.
 *           - Packed storage size must be n*(n+1)/2 (caller guarantees).
 */

#ifndef GSI_SMINRES_LINALG_LAPACK_HPP
#define GSI_SMINRES_LINALG_LAPACK_HPP

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
#include "gsi_sminres/linalg/blas.hpp"

extern "C" {
  // Cholesky 分解
  void zpptrf_(const char* uplo, const blas_int* n,
               std::complex<double>* ap, blas_int *info);
  void zpptrs_(const char* uplo, const blas_int* n, const blas_int* nrhs,
               const std::complex<double> *ap,
               std::complex<double> *b, const blas_int* ldb, blas_int* info);
  // Bunch-Kaufman 分解
  void zhptrf_(const char* uplo, const blas_int* n,
               std::complex<double>* ap, blas_int* ipiv, blas_int* info);
  void zhptrs_(const char* uplo, const blas_int* n, const blas_int* nrhs,
               const std::complex<double>* ap, const blas_int *ipiv,
               std::complex<double>* b, const blas_int* ldb, blas_int* info);
  // Givens rotation
  void zlartg_(const std::complex<double>* f, const std::complex<double>* g,
               double* cs, std::complex<double>* sn, std::complex<double>* r);
}

namespace gsi_sminres {
  namespace linalg {
      /**
       * \namespace gsi_sminres::linalg::lapack
       * \brief [Utility] This namespace provides C++ wrappers for LAPACK routines used in GSI-sMINRES sample programs.
       * \details This namespace provides simple LAPACK routines necessary for
       *          Cholesky factorization and solving linear equations, used in sample programs.
       *
       *          In addition, it provides a wrapper for `zlartg` as a numerically robust
       *          alternative to the BLAS routine `zrotg`, which is known to produce incorrect
       *          results in certain environments such as OpenBLAS versions prior to 0.3.27.
       */
    namespace lapack {

      inline void zpptrf(char uplo, const std::size_t n,
                         std::vector<std::complex<double>>& ap) {
        const blas_int nn = to_blas_int(n);
        blas_int info = 0;
        zpptrf_(&uplo, &nn, ap.data(), &info);
        if (info != 0) {
          throw std::runtime_error("zpptrf failed with info = " + std::to_string(info));
        }
      }

      inline void zpptrs(const char uplo, const std::size_t n,
                         const std::vector<std::complex<double>>& ap,
                         std::vector<std::complex<double>>& x,
                         const std::vector<std::complex<double>>& b) {
        const blas_int nn  = to_blas_int(n);
        const blas_int nr  = 1;
        const blas_int ldb = to_blas_int(n);
        blas_int info = 0;
        blas::zcopy(n, b, 0, x, 0);
        zpptrs_(&uplo, &nn, &nr, ap.data(), x.data(), &ldb, &info);
        if (info != 0) {
          throw std::runtime_error("zpptrs failed with info = " + std::to_string(info));
        }
      }

      /**
       * \brief Perform Bauman-Kaufman factorization of a Hermitian packed matrix.
       * \param[in]     uplo 'U' (upper) or 'L' (lower) — which triangle is stored in \p ap
       * \param[in]     n    order of A
       * \param[in,out] ap   packed A (size n*(n+1)/2), overwritten by the factor (U or L)
       * \param[out]    ipiv
       * \return info (=0: success, <0: -i-th arg illegal, >0: not positive-definite at i)
       */
      inline void zhptrf(char uplo, std::size_t n, std::vector<std::complex<double>>& ap,
                             std::vector<int>& ipiv) {
        const blas_int nn   = to_blas_int(n);
        blas_int info = 0;
        zhptrf_(&uplo, &nn, ap.data(), ipiv.data(), &info);
        if (info != 0) {
          throw std::runtime_error("zhptrf failed with info = " + std::to_string(info));
        }
      }

      /**
       * \brief Solve Ax = b using the Bauman-Kaufman factorization of a Hermitian packed matrix.
       * \param[in]  uplo 'U' or 'L'
       * \param[in]  n    order of A
       * \param[in]  ap   packed factor from ZHPTRF
       * \param[in]  ipiv aaa
       * \param[out] x    solution vector (overwritten). On entry, ignored.
       * \param[in]  b    right-hand side vector (copied into \p x)
       * \return info (=0: success, <0: -i-th arg illegal)
       */
      inline void zhptrs(const char uplo, const std::size_t n,
                             const std::vector<std::complex<double>>& ap,
                             const std::vector<int>& ipiv,
                             std::vector<std::complex<double>>& x,
                             const std::vector<std::complex<double>>& b) {
        const blas_int nn   = to_blas_int(n);
        const blas_int nr   = 1;
        const blas_int ldb  = nn;
        blas_int info = 0;
        blas::zcopy(n, b, 0, x, 0);
        zhptrs_(&uplo, &nn, &nr, ap.data(), ipiv.data(), x.data(), &ldb, &info);
        if (info != 0) {
          throw std::runtime_error("zhptrs failed with info = " + std::to_string(info));
        }
      }

      /**
       * \brief Compute Givens rotation parameters for complex values using LAPACK's `zlartg` routine.
       * \details This function computes the parameters of a Givens rotation matrix
       *          that eliminates the second entry of a 2-vector.
       *          It is a LAPACK-based alternative to zrotg, used in GSMINRES++
       *          to work around known bugs in older versions of OpenBLAS
       *          where zrot does not behave correctly.
       *
       * \param[in,out] f On input, the first component. On output, replaced with the resulting r.
       * \param[in]     g On input, the second component.
       * \param[out]    c The cosine of the rotation.
       * \param[out]    s The sine of the rotation (complex).
       */
      inline void zlartg(std::complex<double>& f, std::complex<double>& g,
                         double& c, std::complex<double>& s){
        std::complex<double> r;
        zlartg_(&f, &g, &c, &s, &r);
        f = r;
      }
      
    }  // namespace lapack
  }  // namespace linalg
}  // namespace gsi_sminres

#endif // GSI_SMINRES_LINALG_LAPACK_HPP
