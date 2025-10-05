/**
 * \file example_gen_zhp.cpp
 * \brief Demo: Generalized shifted MINRES with ZHP storage and LAPACK
 *        packed-Cholesky inner solves.
 * \example example_gen_zhp.cpp
 * \author Shuntaro Hidaka
 *
 * \details
 * Solves \f$(A + \sigma^{(m)} B)x^{(m)} = b\f$ with \c A and \c B read into
 * BLAS packed-Hermitian (ZHP) storage. The inner solves with \f$B^{-1}\f$
 * are performed by \c zpptrf/\c zpptrs (packed-Cholesky), while \c zhpmv
 * applies \c A and \c B.
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
 *   a_gen_zhp.out A.mtx B.mtx
 * \endcode
 *
 * \note Requires BLAS/LAPACK; uses \c zpptrf/\c zpptrs for \f$B^{-1}\f$
 *       and \c zhpmv for matrix-vector products.
 *
 * \see gsi_sminres::generalized::Solver
 * \see gsi_sminres::io::load_mm_zhp
 * \see gsi_sminres::linalg::lapack::zpptrf, gsi_sminres::linalg::lapack::zpptrs
 * \see gsi_sminres::linalg::blas::zhpmv
 */

#include "gsi_sminres/algorithms/generalized_shifted_minres.hpp"
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
  std::vector<std::complex<double>> w(N, {0.0, 0.0}), u(N, {0.0, 0.0});
  std::vector<std::size_t> itr(M);
  std::vector<double>      res(M);

  std::vector<std::complex<double>> Bcholesky = B;
  gsi_sminres::linalg::lapack::zpptrf(uploB, N, Bcholesky);

  gsi_sminres::generalized::Solver solver(N, M);
  gsi_sminres::linalg::lapack::zpptrs(uploB, N, Bcholesky, w, b);
  solver.initialize(x, b, w, sigma, 1e-13);
  for (std::size_t j = 0; j < 10*N; ++j) {
    gsi_sminres::linalg::blas::zhpmv(uploA, N, {1.0,0.0}, A, w, 0, {0.0,0.0}, u, 0);
    solver.glanczos_pre(u);
    gsi_sminres::linalg::lapack::zpptrs(uploB, N, Bcholesky, w, u);
    solver.glanczos_pst(w, u);
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
