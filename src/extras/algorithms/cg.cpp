/**
 * \file cg.cpp
 * \brief Unpreconditioned Conjugate Gradient (CG) implementation.
 */

#include "gsi_sminres/extras/algorithms/cg.hpp"
#include "gsi_sminres/extras/sparse/spmv.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace gsi_sminres {
  namespace extras {

    bool cg(const sparse::CSRMatrix& A,
            std::vector<std::complex<double>>& x,
            const std::vector<std::complex<double>>& b,
            std::size_t max_iter, double rtol,
            std::size_t* iters, double* relres) {
      std::size_t N = A.n;
      if (x.size() != N || b.size() != N) {
        throw std::invalid_argument("cg: size mismatch between A, x, b.");
      }

      std::size_t i = 0;
      std::vector<std::complex<double>> p(N), r(N), Ap(N);
      double alpha, beta, rr_curr, rr_next;
      double rres;
      double bnrm = linalg::blas::dznrm2(N, b);

      sparse::SpMV(A, x, Ap);
      linalg::blas::zcopy(N, b, 0, r, 0);
      linalg::blas::zaxpy(N, {-1.0,0.0}, Ap, 0, r, 0);
      rr_curr = std::real(linalg::blas::zdotc(N, r, 0, r, 0));
      rres = std::sqrt(rr_curr)/bnrm;
      if (rres <= rtol) {
        if (iters)  *iters  = 0;
        if (relres) *relres = rres;
        return true;
      }
      linalg::blas::zcopy(N, r, 0, p, 0);
      for (i = 0; i < max_iter; ++i) {
        sparse::SpMV(A, p, Ap);
        alpha = rr_curr / std::real(linalg::blas::zdotc(N, p, 0, Ap, 0));
        linalg::blas::zaxpy(N, alpha,   p, 0, x, 0);
        linalg::blas::zaxpy(N, -alpha, Ap, 0, r, 0);
        rr_next = std::real(linalg::blas::zdotc(N, r, 0, r, 0));
        rres = std::sqrt(rr_next)/bnrm;
        if (rres <= rtol) {
          if (iters)  *iters  = i + 1;
          if (relres) *relres = rres;
          return true;
        }
        beta = rr_next / rr_curr;
        linalg::blas::zdscal(N, beta, p);
        linalg::blas::zaxpy(N, {1.0,0.0}, r, 0, p, 0);
        rr_curr = rr_next;
      }

      if (iters)  *iters  = i;
      if (relres) *relres = rres;
      return false;
    }

    bool cg_r(const sparse::CSRMatrix_r& A,
              std::vector<double>& x,
              const std::vector<double>& b,
              std::size_t max_iter, double rtol,
              std::size_t* iters, double* relres) {
      std::size_t N = A.n;
      if (x.size() != N || b.size() != N) {
        throw std::invalid_argument("cg_r: size mismatch between A, x, b.");
      }

      std::size_t i = 0;
      std::vector<double> p(N), r(N), Ap(N);
      double alpha, beta, rr_curr, rr_next;
      double rres;
      double bnrm = linalg::blas::dnrm2(N, b);

      sparse::SpMV_r(A, x, Ap);
      linalg::blas::dcopy(N, b, 0, r, 0);
      linalg::blas::daxpy(N, -1.0, Ap, 0, r, 0);
      rr_curr = linalg::blas::ddot(N, r, 0, r, 0);
      rres = std::sqrt(rr_curr)/bnrm;
      if (rres <= rtol) {
        if (iters)  *iters  = 0;
        if (relres) *relres = rres;
        return true;
      }
      linalg::blas::dcopy(N, r, 0, p, 0);
      for (i = 0; i < max_iter; ++i) {
        sparse::SpMV_r(A, p, Ap);
        alpha = rr_curr / linalg::blas::ddot(N, p, 0, Ap, 0);
        linalg::blas::daxpy(N, alpha,   p, 0, x, 0);
        linalg::blas::daxpy(N, -alpha, Ap, 0, r, 0);
        rr_next = linalg::blas::ddot(N, r, 0, r, 0);
        rres = std::sqrt(rr_next)/bnrm;
        if (rres <= rtol) {
          if (iters)  *iters  = i + 1;
          if (relres) *relres = rres;
          return true;
        }
        beta = rr_next / rr_curr;
        linalg::blas::dscal(N, beta, p);
        linalg::blas::daxpy(N, 1.0, r, 0, p, 0);
        rr_curr = rr_next;
      }

      if (iters)  *iters  = i;
      if (relres) *relres = rres;
      return false;
    }

  }  // namespace extras
}  // namespace gsi_sminres
