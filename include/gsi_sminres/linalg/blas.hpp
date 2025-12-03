/**
 * \file blas.hpp
 * \brief BLAS wrappers for the GSI-MINRES solver.
 * \author Shuntaro Hidaka
 *
 * \details This header defines lightweight C++ wrapper functions for selected BLAS Level-1 routines
 *          such as `zaxpy`, `dznrm2` and `zdotc` which are used internally in GSI-sMINRES.
 *          The interfaces are designed for safety and usability using `std::vector`,
 *          and provide explicit control over starting offsets and memory strides
 *          for advanced vector operations.
 */

#ifndef GSI_SMINRES_LINALG_BLAS_HPP
#define GSI_SMINRES_LINALG_BLAS_HPP

#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits> // std::conditional_t
#include <vector>

// BLAS and integer width switching
#if defined(USE_MKL)
  #include <mkl.h>
  using blas_int = MKL_INT;
#elif defined(USE_OPENBLAS)
  #include <cblas.h>
  using blas_int = blasint;
#else
  #include <cblas.h>
  using blas_int = int;
#endif

// size_t to blas_int casting function
// デバッグオプションがあるときだけエラーを出したい、要検討
#if defined(NDEBUG)
  static_assert(std::is_signed_v<blas_int>, "blas_int must be signed");
  static_assert(sizeof(blas_int) == 4 || sizeof(blas_int) == 8, "unexpected blas_int width");
  struct [[deprecated("BLAS: blas_int is narrower than std::size_t;"
                      "ensure values fit or build ILP64")]] _Bad{};
  struct _Ok{};
  [[maybe_unused]] inline
  std::conditional_t<(sizeof(blas_int) < sizeof(std::size_t)), _Bad, _Ok> _blas_size_warning{};
#endif
constexpr inline blas_int to_blas_int(std::size_t val) noexcept {
  return static_cast<blas_int>(val);
}

extern "C" {
  void zrotg_(std::complex<double> *a, std::complex<double> *b,
              double *c, std::complex<double> *s);
  void zrot_(const blas_int *n,
             std::complex<double> *x, const blas_int *incx,
             std::complex<double> *y, const blas_int *incy,
             const double *c, const std::complex<double> *s);
}

namespace gsi_sminres {
  namespace linalg {
    /**
     * \namespace gsi_sminres::linalg::blas
     * \brief [Utility] This namespace provides C++ wrappers for BLAS routines used in GSI-sMINRES sample programs.
     * \details This namespace provides simple C++ wrappers over the BLAS routines.
     *          These routines are used in GSI-MINRES.
     */
    namespace blas {
      // ========== Level-1: scal ==========
      /**
       * \brief Scale a real vector \f$ x \f$ by a real scalar \f$ \alpha \f$.
       * \param[in]     n        Number of elements to scale.
       * \param[in]     alpha    Real scalar multiplier.
       * \param[in,out] x        Real vector to scale.
       * \param[in]     x_offset Starting index within the x vector (zero-based offset).
       * \param[in]     incx     Step size between elements in the x vector (stride).
       */
      inline void dscal(std::size_t n, double alpha, std::vector<double>& x,
                        std::size_t x_offset=0, std::size_t incx=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx);
        cblas_dscal(nn, alpha, x.data()+x_offset, ix);
      }
      /**
       * \brief Scale a complex vector \f$ x \f$ by a real scalar \f$ \alpha \f$.
       * \param[in]     n        Number of elements to scale.
       * \param[in]     alpha    Real scalar multiplier.
       * \param[in,out] x        Complex vector to scale.
       * \param[in]     x_offset Starting index within the x vector.
       * \param[in]     incx     Step size between elements in the x vector.
       */
      inline void zdscal(std::size_t n, double alpha, std::vector<std::complex<double>>& x,
                         std::size_t x_offset=0, std::size_t incx=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx);
        cblas_zdscal(nn, alpha, x.data()+x_offset, ix);
      }
      /**
       * \brief Scale a complex vector \f$ x \f$ by a complex scalar \f$ \alpha \f$.
       * \param[in]     n        Number of elements to scale.
       * \param[in]     alpha    Complex scalar multiplier.
       * \param[in,out] x        Complex vector to scale.
       * \param[in]     x_offset Starting index within the x vector.
       * \param[in]     incx     Step size between elements in the x vector.
       */
      inline void zscal(std::size_t n, std::complex<double> alpha, std::vector<std::complex<double>>& x,
                        std::size_t x_offset=0, std::size_t incx=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx);
        cblas_zscal(nn, &alpha, x.data()+x_offset, ix);
      }

      // ========== Level-1: copy ==========
      /**
       * \brief Copy real vector \f$ x \f$ into real vector \f$ y \f$.
       * \param[in]  n        Number of elements to copy.
       * \param[in]  x        Real source vector.
       * \param[in]  x_offset Starting index within the x vector.
       * \param[out] y        Real destination vector.
       * \param[in]  y_offset Starting index within the y vector.
       * \param[in]  incx     Step size between elements in the x vector.
       * \param[in]  incy     Step size between elements in the y vector.
       */
      inline void dcopy(std::size_t n,
                        const std::vector<double>& x, std::size_t x_offset,
                        std::vector<double>&       y, std::size_t y_offset,
                        std::size_t incx=1, std::size_t incy=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx), iy = to_blas_int(incy);
        cblas_dcopy(nn, x.data()+x_offset, ix, y.data()+y_offset, iy);
      }
      /**
       * \brief Copy complex vector \f$ x \f$ into complex vector \f$ y \f$.
       * \param[in]  n        Number of elements to copy.
       * \param[in]  x        Complex source vector.
       * \param[in]  x_offset Starting index within the x vector.
       * \param[out] y        Complex destination vector.
       * \param[in]  y_offset Starting index within the y vector.
       * \param[in]  incx     Step size between elements in the x vector.
       * \param[in]  incy     Step size between elements in the y vector.
       */
      inline void zcopy(std::size_t n,
                        const std::vector<std::complex<double>>& x, std::size_t x_offset,
                        std::vector<std::complex<double>>&       y, std::size_t y_offset,
                        std::size_t incx=1, std::size_t incy=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx), iy = to_blas_int(incy);
        cblas_zcopy(nn, x.data()+x_offset, ix, y.data()+y_offset, iy);
      }

      // ========== Level-1: axpy ==========
      /**
       * \brief Perform \f$ y \leftarrow \alpha x + y \f$ for Real vector
       * \param[in]     n        Number of elements to perform (length).
       * \param[in]     alpha    Scalar multiplier.
       * \param[in]     x        Input vector.
       * \param[in]     x_offset Starting index within the x vector.
       * \param[in,out] y        Output vector (accumulated).
       * \param[in]     y_offset Starting index within the y vector.
       * \param[in]     incx     Step size between elements in the x vector.
       * \param[in]     incy     Step size between elements in the y vector.
       */
      inline void daxpy(std::size_t n, double alpha,
                        const std::vector<double>& x, std::size_t x_offset,
                        std::vector<double>&       y, std::size_t y_offset,
                        std::size_t incx=1, std::size_t incy=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx), iy = to_blas_int(incy);
        cblas_daxpy(nn, alpha, x.data()+x_offset, ix, y.data()+y_offset, iy);
      }
      /**
       * \brief Perform \f$ y \leftarrow \alpha x + y \f$ for complex vector
       * \param[in]     n        Number of elements to perform (length).
       * \param[in]     alpha    Scalar multiplier.
       * \param[in]     x        Input vector.
       * \param[in]     x_offset Starting index within the x vector.
       * \param[in,out] y        Output vector (accumulated).
       * \param[in]     y_offset Starting index within the y vector.
       * \param[in]     incx     Step size between elements in the x vector.
       * \param[in]     incy     Step size between elements in the y vector.
       */
      inline void zaxpy(std::size_t n, std::complex<double> alpha,
                        const std::vector<std::complex<double>>& x, std::size_t x_offset,
                        std::vector<std::complex<double>>&       y, std::size_t y_offset,
                        std::size_t incx=1, std::size_t incy=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx), iy = to_blas_int(incy);
        cblas_zaxpy(nn, &alpha, x.data()+x_offset, ix, y.data()+y_offset, iy);
      }

      // ========== Level-1: dot ==========
      /**
       * \brief Compute dot product of Real vectors: \f$ \sum_{i=0}^{n-1} x_i  y_i \f$.
       * \param[in] n        Number of elements to compute.
       * \param[in] x        First input vector.
       * \param[in] x_offset Starting index within the x vector.
       * \param[in] y        Second input vector.
       * \param[in] y_offset Starting index within the y vector.
       * \param[in]  incx    Step size between elements in the x vector.
       * \param[in]  incy    Step size between elements in the y vector.
       * \return Real scalar result.
       */
      inline double ddot(std::size_t n,
                         const std::vector<double>& x, std::size_t x_offset,
                         const std::vector<double>& y, std::size_t y_offset,
                         std::size_t incx=1, std::size_t incy=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx), iy = to_blas_int(incy);
        return cblas_ddot(nn, x.data()+x_offset, ix, y.data()+y_offset, iy);
      }
      /**
       * \brief Compute dot product of complex vectors: \f$ \sum_{i=0}^{n-1} \overline{x_i}  y_i \f$.
       * \param[in] n        Number of elements to compute.
       * \param[in] x        First input vector.
       * \param[in] x_offset Starting index within the x vector.
       * \param[in] y        Second input vector.
       * \param[in] y_offset Starting index within the y vector.
       * \param[in]  incx    Step size between elements in the x vector.
       * \param[in]  incy    Step size between elements in the y vector.
       * \return Complex scalar result.
       */
      inline std::complex<double> zdotc(std::size_t n,
                                        const std::vector<std::complex<double>>& x, std::size_t x_offset,
                                        const std::vector<std::complex<double>>& y, std::size_t y_offset,
                                        std::size_t incx=1, std::size_t incy=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx), iy = to_blas_int(incy);
        std::complex<double> r;
        cblas_zdotc_sub(nn, x.data()+x_offset, ix, y.data()+y_offset, iy, &r);
        return r;
      }

      // ========== Level-1: nrm2 ==========
      /**
       * \brief Compute the Euclidean norm (2-norm) of a real vector: \f$ \|x\|_2 \f$.
       * \param[in] n        Number of elements to compute.
       * \param[in] x        Input vector.
       * \param[in] x_offset Starting index within the x vector.
       * \param[in] incx     Step size between elements in the x vector.
       * \return 2-norm value (double).
       */
      inline double dnrm2(std::size_t n,
                           const std::vector<double>& x, std::size_t x_offset=0,
                           std::size_t incx=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx);
        return cblas_dnrm2(nn, x.data()+x_offset, ix);
      }
      /**
       * \brief Compute the Euclidean norm (2-norm) of a complex vector: \f$ \|x\|_2 \f$.
       * \param[in] n        Number of elements to compute.
       * \param[in] x        Input vector.
       * \param[in] x_offset Starting index within the x vector.
       * \param[in] incx     Step size between elements in the x vector.
       * \return 2-norm value (double).
       */
      inline double dznrm2(std::size_t n,
                           const std::vector<std::complex<double>>& x, std::size_t x_offset=0,
                           std::size_t incx=1) noexcept {
        const blas_int nn = to_blas_int(n);
        const blas_int ix = to_blas_int(incx);
        return cblas_dznrm2(nn, x.data()+x_offset, ix);
      }

      // ========== Level-1: rot ==========
      /**
       * \brief Compute Givens rotation parameters.
       * \param[in,out] a First component, overwritten.
       * \param[in,out] b Second component, overwritten.
       * \param[out]    c Cosine value (real).
       * \param[out]    s Sine value (complex).
       */
      inline void zrotg(std::complex<double>& a, std::complex<double>& b,
                        double& c, std::complex<double>& s) noexcept {
        zrotg_(&a, &b, &c, &s);
      }

      // ========== Level-1 rotg ==========
      /**
       * \brief Apply Givens rotation to vector pair (x, y).
       * \param[in]     n        Number of elements to apply.
       * \param[in,out] x        First vector, overwritten c*x+s*y.
       * \param[in]     x_offset Starting index within the x vector.
       * \param[in,out] y        Second vector, overwritten -conj(s)*x+c*y.
       * \param[in]     y_offset Starting index within the y vector.
       * \param[in]     c        Cosine value.
       * \param[in]     s        Sine value (complex).
       * \param[in]     incx     Step size between elements in the x vector.
       * \param[in]     incy     Step size between elements in the y vector.
       */
      inline void zrot(std::size_t n,
                       std::vector<std::complex<double>>& x, std::size_t x_offset,
                       std::vector<std::complex<double>>& y, std::size_t y_offset,
                       double c, std::complex<double> s,
                       std::size_t incx=1, std::size_t incy=1) noexcept {
        blas_int nn = to_blas_int(n);
        blas_int ix = to_blas_int(incx), iy = to_blas_int(incy);
        zrot_(&nn, x.data()+x_offset, &ix, y.data()+y_offset, &iy, &c, &s);
      }

    }  // namespace blas
  }  // namespace linalg
}  // namespace gsi_sminres


#endif // GSI_SMINRES_LINALG_BLAS_HPP
