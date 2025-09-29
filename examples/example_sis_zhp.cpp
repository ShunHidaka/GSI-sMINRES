/**
 * \file example_sis_zhp.cpp
 * \brief Demo: Shift–Invert preconditioned shifted MINRES with BLAS/LAPACK
 *        packed-Hermitian (ZHP) backends.
 *
 * \details
 * Solves \f$(A + \sigma^{(m)} B)x^{(m)} = b\f$ using the Shift–Invert
 * preconditioned shifted MINRES solver. Both \c A and \c B are read into
 * BLAS/LAPACK packed-Hermitian (ZHP) storage; \f$S = A + \omega B\f$ is
 * factorized by \c zhptrf and applied via \c zhptrs every outer iteration.
 *
 * **Input**
 *  - `A.mtx` (Hermitian / symmetric, ZHP after loading)
 *  - `B.mtx` (Hermitian positive definite, ZHP after loading)
 *  - The program enforces `uploA == uploB` to keep storage consistent.
 *
 * **Output (per shift m)**
 *  - index, Re(σ), Im(σ), iteration count, algorithmic residual estimate,
 *    and the true 2-norm residual \f$\|(A+\sigma B)x-b\|_2\f$.
 *
 * \par Usage
 * \code
 *   a_sis_zhp.out A.mtx B.mtx
 * \endcode
 *
 * \note Requires BLAS/LAPACK; uses \c zhpmv for Hermitian MVs and
 *       \c zhptrf/\c zhptrs for solving with \f$A + \omega B\f$.
 *
 * \see gsi_sminres::shift_invert::Solver
 * \see gsi_sminres::io::load_mm_zhp
 * \see gsi_sminres::linalg::blas::zhpmv
 * \see gsi_sminres::linalg::lapack::zhptrf, gsi_sminres::linalg::lapack::zhptrs
 */


#include "gsi_sminres/algorithms/shift_invert_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include "gsi_sminres/linalg/blas_zhpmv.hpp"
#include "gsi_sminres/linalg/lapack.hpp"
#include "gsi_sminres/extras/io/mm_zhp.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <vector>

int main(int argc, char* argv[]) {
  // Matrix size, Shift size
  std::size_t N, M=10;

  // Load matrix
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <MTX_file(A)> <MTX_file(B)>" << std::endl;
    return 1;
  }
  std::string Aname = argv[1], Bname = argv[2];
  char uploA, uploB;
  const auto A = gsi_sminres::io::load_mm_zhp(Aname, N, uploA);
  const auto B = gsi_sminres::io::load_mm_zhp(Bname, N, uploB);
  // Prepare shifts
  std::vector<std::complex<double>> sigma(M);
  for (std::size_t m = 0; m < M; ++m) {
    sigma[m] = std::polar(0.01, 2*std::acos(-1)*(m+0.5)/M);
  }
  // Prepare rhs
  std::vector<std::complex<double>> b(N, {1.0, 0.0});

  // Prepare variables
  std::vector<std::complex<double>> x(M*N, {0.0, 0.0});
  std::vector<std::complex<double>> v(N, {0.0, 0.0}), Bv(N, {0.0, 0.0});
  std::vector<std::size_t> itr(M);
  std::vector<double>      res(M);

  // Constructing shift-invert preconditioner
  const std::complex<double> omega = 0.0;
  const char uploS = [&]() -> char {
    if (uploA != uploB)
      throw std::runtime_error("uploA != uploB.");
    return uploA;
  }();
  std::vector<int> ipiv(N);
  std::vector<std::complex<double>> S = A;
  gsi_sminres::linalg::blas::zaxpy(S.size(), omega, B, 0, S, 0);
  gsi_sminres::linalg::lapack::zhptrf(uploS, N, S, ipiv);

  gsi_sminres::shift_invert::Solver solver(N, M);
  gsi_sminres::linalg::lapack::zhptrs(uploS, N, S, ipiv, v, b);
  gsi_sminres::linalg::blas::zhpmv(uploB, N, {1.0,0.0}, B, v, 0, {0.0,0.0}, Bv, 0);
  solver.initialize(x, v, Bv, sigma, omega, 1e-12);
  for (std::size_t j = 0; j < N; ++j) {
    gsi_sminres::linalg::lapack::zhptrs(uploS, N, S, ipiv, v, Bv);
    solver.sislanczos_pre(v, Bv);
    gsi_sminres::linalg::blas::zhpmv(uploB, N, {1.0,0.0}, B, v, 0, {0.0,0.0}, Bv, 0);
    solver.sislanczos_pst(v, Bv);
    if (solver.update(x)) {
      break;
    }
    solver.get_alg_residual(res);
  }
  solver.finalize(itr, res);

  // Output results
  for (std::size_t m = 0; m < M; ++m) {
    std::vector<std::complex<double>> ans(x.begin()+m*N, x.begin()+(m+1)*N);
    std::vector<std::complex<double>> tmp(N, {0.0, 0.0});
    gsi_sminres::linalg::blas::zhpmv(uploA, N, {1.0,0.0}, A, ans, 0, {0.0,0.0}, tmp, 0);
    gsi_sminres::linalg::blas::zhpmv(uploB, N,  sigma[m], B, ans, 0, {1.0,0.0}, tmp, 0);
    gsi_sminres::linalg::blas::zaxpy(N, {-1.0, 0.0}, b,   0, tmp, 0);
    double tmp_nrm = gsi_sminres::linalg::blas::dznrm2(N, tmp);
    std::cout << std::right
              << std::setw(2) << m << " "
              << std::fixed << std::setw(10) << std::setprecision(6) << sigma[m].real() << " "
              << std::fixed << std::setw(10) << std::setprecision(6) << sigma[m].imag() << " "
              << std::setw(5) << itr[m] << " "
              << std::scientific << std::setw(12) << std::setprecision(5) << res[m] << " "
              << std::scientific << std::setw(12) << std::setprecision(5) << tmp_nrm
              << std::endl;
  }

  return 0;
}
