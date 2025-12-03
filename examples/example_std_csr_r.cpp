/**
 * \file example_std_csr.cpp
 * \brief Demo: Standard shifted MINRES (no B) with CSR SpMV.
 * \example example_std_csr.cpp
 * \author Shuntaro Hidaka
 *
 * \details
 * Solves a family of shifted systems \f$(A + \sigma^{(m)} I)x^{(m)} = b\f$
 * by the standard shifted MINRES method. Matrix \c A is loaded in CSR
 * and applied through a simple SpMV; vector ops use BLAS Level-1.
 *
 * **Input**
 *  - `A.mtx` (Hermitian / real-symmetric, CSR after loading)
 *
 * **Output (per shift m)**
 *  - index, Re(σ), Im(σ), iteration count, algorithmic residual estimate,
 *    and the true residual norm \f$\|(A+\sigma I)x-b\|_2\f$.
 *
 * \par Usage
 * \code
 *   a_std_csr.out A.mtx
 * \endcode
 *
 * \note The solution array \c x stores M blocks of length N (one per shift).
 *
 * \see gsi_sminres::standard::Solver
 * \see gsi_sminres::io::load_mm_csr
 * \see gsi_sminres::sparse::SpMV
 */

#include "gsi_sminres/algorithms/standard_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include "gsi_sminres/extras/sparse/csr.hpp"
#include "gsi_sminres/extras/sparse/spmv.hpp"
#include "gsi_sminres/extras/io/mm_csr.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
  // Matrix size, Shift size
  std::size_t N, M=10;

  // Load matrix
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <MTX_file(A)>" << std::endl;
    return 1;
  }
  std::string Aname = argv[1];
  const auto A = gsi_sminres::io::load_mm_csr_r(Aname, N);
  // Prepare shifts
  std::vector<std::complex<double>> sigma(M);
  for (std::size_t m = 0; m < M; ++m) {
    sigma[m] = std::polar(0.01, 2*std::acos(-1)*(m+0.5)/M);
  }
  // Prepare rhs
  std::vector<double> b(N, 1.0);

  // Prepare variables
  std::vector<std::complex<double>> x(M*N, {0.0, 0.0});
  std::vector<double> v(N, 0.0), Av(N, 0.0);
  std::vector<std::size_t> itr(M);
  std::vector<double>      res(M);

  auto start = std::chrono::high_resolution_clock::now();

  gsi_sminres::standard::Solver solver(N, M);
  solver.initialize_r(x, b, v, sigma, 1e-12);
  for (std::size_t j = 0; j < 10*N; ++j) {
    gsi_sminres::sparse::SpMV_r(A, v, Av);
    solver.lanczos_r(v, Av);
    if (solver.update(x)) {
      break;
    }
    solver.get_alg_residual(res);
  }
  solver.finalize(itr, res);

  auto end = std::chrono::high_resolution_clock::now();

  // Output results
  double sec = std::chrono::duration<double>(end - start).count();
  std::cout << "# time = " << sec << " s" << std::endl;
  for (std::size_t m = 0; m < M; ++m) {
    const auto A_cmplx = gsi_sminres::io::load_mm_csr(Aname, N);
    std::vector<std::complex<double>> b_cmplx(N, {1.0, 0.0});
    std::vector<std::complex<double>> ans(x.begin()+m*N, x.begin()+(m+1)*N);
    std::vector<std::complex<double>> tmp(N, {0.0, 0.0});
    gsi_sminres::sparse::SpMV(A_cmplx, ans, tmp);
    gsi_sminres::linalg::blas::zaxpy(N,    sigma[m], ans, 0, tmp, 0);
    gsi_sminres::linalg::blas::zaxpy(N, {-1.0, 0.0}, b_cmplx,   0, tmp, 0);
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
