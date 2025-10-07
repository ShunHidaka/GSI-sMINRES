/**
 * \file example_std_zhp.cpp
 * \brief Demo: Standard shifted MINRES (no B) with BLAS packed-Hermitian MVs.
 * \example example_std_zhp.cpp
 * \author Shuntaro Hidaka
 *
 * \details
 * Solves \f$(A + \sigma^{(m)} I)x^{(m)} = b\f$ using the standard shifted
 * MINRES algorithm. Matrix \c A is read into BLAS packed-Hermitian (ZHP)
 * storage and applied by \c zhpmv each iteration.
 *
 * **Input**
 *  - `A.mtx` (Hermitian / symmetric, ZHP after loading)
 *
 * **Output (per shift m)**
 *  - index, Re(σ), Im(σ), iteration count, algorithmic residual estimate,
 *    and the true residual norm \f$\|(A+\sigma I)x-b\|_2\f$.
 *
 * \par Usage
 * \code
 *   a_std_zhp.out A.mtx
 * \endcode
 *
 * \see gsi_sminres::standard::Solver
 * \see gsi_sminres::io::load_mm_zhp
 * \see gsi_sminres::linalg::blas::zhpmv
 */

#include "gsi_sminres/algorithms/standard_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include "gsi_sminres/linalg/blas_zhpmv.hpp"
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
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <MTX_file(A)>" << std::endl;
    return 1;
  }
  std::string Aname = argv[1];
  char uploA;
  const auto A = gsi_sminres::io::load_mm_zhp(Aname, N, uploA);
  // Prepare shifts
  std::vector<std::complex<double>> sigma(M);
  for (std::size_t m = 0; m < M; ++m) {
    sigma[m] = std::polar(0.01, 2*std::acos(-1)*(m+0.5)/M);
  }
  // Prepare rhs
  std::vector<std::complex<double>> b(N, {1.0, 0.0});

  // Prepare variables
  std::vector<std::complex<double>> x(M*N, {0.0, 0.0});
  std::vector<std::complex<double>> v(N, {0.0, 0.0}), Av(N, {0.0, 0.0});
  std::vector<std::size_t> itr(M);
  std::vector<double>      res(M);

  gsi_sminres::standard::Solver solver(N, M);
  solver.initialize(x, b, v, sigma, 1e-12);
  for (std::size_t j = 0; j < 10*N; ++j) {
    gsi_sminres::linalg::blas::zhpmv(uploA, N, {1.0,0.0}, A, v, 0, {0.0,0.0}, Av, 0);
    solver.lanczos(v, Av);
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
    gsi_sminres::linalg::blas::zaxpy(N,    sigma[m], ans, 0, tmp, 0);
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
