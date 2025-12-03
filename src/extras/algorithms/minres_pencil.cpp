/**
 * \file minres_pencil.cpp
 * \brief Unpreconditioned MINRES implementation for the \f$ (A + \omega B) \f$ pencil.
 */

#include "gsi_sminres/extras/algorithms/minres_pencil.hpp"
#include "gsi_sminres/extras/sparse/csr.hpp"
#include "gsi_sminres/extras/sparse/spmv.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace gsi_sminres {
  namespace extras {

    bool minres_pencil(const sparse::CSRMatrix&                 A,
                       const std::complex<double>               omega,
                       const sparse::CSRMatrix&                 B,
                       std::vector<std::complex<double>>&       x,
                       const std::vector<std::complex<double>>& b,
                       std::size_t                              max_iter,
                       double                                   rtol,
                       std::size_t*                             iters,
                       double*                                  relres) {
      std::size_t N = A.n;
      if (B.n != N || x.size() != N || b.size() != N) {
        throw std::invalid_argument("minres_pencil: size mismatch between A, B, x, b.");
      }

      std::size_t i = 0;
      double alpha, beta_prev, beta_curr;
      std::vector<std::complex<double>> v_prev(N), v_curr(N), v_next(N);
      std::vector<std::complex<double>> T_prev2(1), T_prev(1), T_curr(1), T_next(1);
      std::vector<double>               Gc(3);
      std::vector<std::complex<double>> Gs(3);
      std::vector<std::complex<double>> p_prev2(N), p_prev(N), p_curr(N);
      std::complex<double> f({1.0, 0.0});
      double h;
      double rres, r0nrm, bnrm = linalg::blas::dznrm2(N, b);
      std::vector<std::complex<double>> Bv_curr(N);

      linalg::blas::zcopy(N, b, 0, v_curr, 0);
      sparse::SpMV(A, x, v_next); // v_nextはとりあえず使用しているだけ
      sparse::SpMV(B, x, Bv_curr);
      linalg::blas::zaxpy(N, omega, Bv_curr, 0, v_next, 0);
      linalg::blas::zaxpy(N, {-1.0,0.0}, v_next, 0, v_curr, 0); // v_curr <- b - (A + \omega B) x
      r0nrm = linalg::blas::dznrm2(N, v_curr);
      h = r0nrm;
      rres = h / bnrm;
      if (rres <= rtol) {
        if (iters)  *iters  = 0;
        if (relres) *relres = rres;
        return true;
      }
      beta_prev = 0.0;
      linalg::blas::zdscal(N, 1.0/h, v_curr);
      for (i = 0; i < max_iter; ++i) {
        // Lanczos process for (A + \omega B)
        sparse::SpMV(A, v_curr, v_next);
        sparse::SpMV(B, v_curr, Bv_curr);
        linalg::blas::zaxpy(N, omega, Bv_curr, 0, v_next, 0);
        alpha = std::real(linalg::blas::zdotc(N, v_curr, 0, v_next, 0));
        linalg::blas::zaxpy(N, -beta_prev, v_prev, 0, v_next, 0);
        linalg::blas::zaxpy(N, -alpha,     v_curr, 0, v_next, 0);
        beta_curr = linalg::blas::dznrm2(N, v_next);
        linalg::blas::zdscal(N, 1.0/beta_curr, v_next);
        // Minimize residual solution
        T_prev2[0] = 0.0;
        T_prev[0] = beta_prev; T_curr[0] = alpha; T_next[0] = beta_curr;
        if (i >= 2) {
          linalg::blas::zrot(1, T_prev2, 0, T_prev, 0, Gc[0], Gs[0]);
        }
        if (i >= 1) {
          linalg::blas::zrot(1, T_prev,  0, T_curr, 0, Gc[1], Gs[1]);
        }
        linalg::blas::zrotg(T_curr[0], T_next[0], Gc[2], Gs[2]);
        std::swap(p_prev, p_prev2);
        std::swap(p_curr, p_prev);
        linalg::blas::zcopy(N, v_curr, 0, p_curr, 0);
        linalg::blas::zaxpy(N, -T_prev2[0], p_prev2, 0, p_curr, 0);
        linalg::blas::zaxpy(N, -T_prev[0],  p_prev,  0, p_curr, 0);
        linalg::blas::zscal(N, 1.0/T_curr[0], p_curr);
        linalg::blas::zaxpy(N, r0nrm*Gc[2]*f, p_curr, 0, x, 0);
        f = -std::conj(Gs[2]) * f;
        h = std::abs(-std::conj(Gs[2])) * h;
        rres = h / bnrm;
        if (rres <= rtol) {
          if (iters)  *iters  = i + 1;
          if (relres) *relres = rres;
          return true;
        }
        Gc[0] = Gc[1]; Gc[1] = Gc[2];
        Gs[0] = Gs[1]; Gs[1] = Gs[2];
        beta_prev = beta_curr;
        std::swap(v_curr, v_prev);
        std::swap(v_next, v_curr);
      }

      if (iters)  *iters  = i;
      if (relres) *relres = rres;
      return false;
    }

    bool minres_pencil_r(const sparse::CSRMatrix_r& A,
                         const double               omega,
                         const sparse::CSRMatrix_r& B,
                         std::vector<double>&       x,
                         const std::vector<double>& b,
                         std::size_t                max_iter,
                         double                     rtol,
                         std::size_t*               iters,
                         double*                    relres) {
      std::size_t N = A.n;
      if (B.n != N || x.size() != N || b.size() != N) {
        throw std::invalid_argument("minres_pencil: size mismatch between A, B, x, b.");
      }

      std::size_t i = 0;
      double alpha, beta_prev, beta_curr;
      std::vector<double> v_prev(N), v_curr(N), v_next(N);
      std::vector<double> T_prev2(1), T_prev(1), T_curr(1), T_next(1);
      std::vector<double> Gc(3);
      std::vector<double> Gs(3);
      std::vector<double> p_prev2(N), p_prev(N), p_curr(N);
      double f = 1.0;
      double h;
      double rres, r0nrm, bnrm = linalg::blas::dnrm2(N, b);
      std::vector<double> Bv_curr(N);

      linalg::blas::dcopy(N, b, 0, v_curr, 0);
      sparse::SpMV_r(A, x, v_next); // v_nextはとりあえず使用しているだけ
      sparse::SpMV_r(B, x, Bv_curr);
      linalg::blas::daxpy(N, omega, Bv_curr, 0, v_next, 0);
      linalg::blas::daxpy(N, -1.0,  v_next,  0, v_curr, 0); // v_curr <- b - (A + \omega B) x
      r0nrm = linalg::blas::dnrm2(N, v_curr);
      h = r0nrm;
      rres = h / bnrm;
      if (rres <= rtol) {
        if (iters)  *iters  = 0;
        if (relres) *relres = rres;
        return true;
      }
      beta_prev = 0.0;
      linalg::blas::dscal(N, 1.0/h, v_curr);
      for (i = 0; i < max_iter; ++i) {
        // Lanczos process for (A + \omega B)
        sparse::SpMV_r(A, v_curr, v_next);
        sparse::SpMV_r(B, v_curr, Bv_curr);
        linalg::blas::daxpy(N, omega, Bv_curr, 0, v_next, 0);
        alpha = linalg::blas::ddot(N, v_curr, 0, v_next, 0);
        linalg::blas::daxpy(N, -beta_prev, v_prev, 0, v_next, 0);
        linalg::blas::daxpy(N, -alpha,     v_curr, 0, v_next, 0);
        beta_curr = linalg::blas::dnrm2(N, v_next);
        linalg::blas::dscal(N, 1.0/beta_curr, v_next);
        // Minimize residual solution
        T_prev2[0] = 0.0;
        T_prev[0] = beta_prev; T_curr[0] = alpha; T_next[0] = beta_curr;
        if (i >= 2) {
          linalg::blas::drot(1, T_prev2, 0, T_prev, 0, Gc[0], Gs[0]);
        }
        if (i >= 1) {
          linalg::blas::drot(1, T_prev,  0, T_curr, 0, Gc[1], Gs[1]);
        }
        linalg::blas::drotg(T_curr[0], T_next[0], Gc[2], Gs[2]);
        std::swap(p_prev, p_prev2);
        std::swap(p_curr, p_prev);
        linalg::blas::dcopy(N, v_curr, 0, p_curr, 0);
        linalg::blas::daxpy(N, -T_prev2[0], p_prev2, 0, p_curr, 0);
        linalg::blas::daxpy(N, -T_prev[0],  p_prev,  0, p_curr, 0);
        linalg::blas::dscal(N, 1.0/T_curr[0], p_curr);
        linalg::blas::daxpy(N, r0nrm*Gc[2]*f, p_curr, 0, x, 0);
        f = -Gs[2] * f;
        h = std::abs(Gs[2]) * h;
        rres = h / bnrm;
        if (rres <= rtol) {
          if (iters)  *iters  = i + 1;
          if (relres) *relres = rres;
          return true;
        }
        Gc[0] = Gc[1]; Gc[1] = Gc[2];
        Gs[0] = Gs[1]; Gs[1] = Gs[2];
        beta_prev = beta_curr;
        std::swap(v_curr, v_prev);
        std::swap(v_next, v_curr);
      }

      if (iters)  *iters  = i;
      if (relres) *relres = rres;
      return false;
    }

  }  // namespace extras
}  // namespace gsi_sminres
