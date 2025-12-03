/**
 * \file example_gen_csr.cpp
 * \brief Demo: Generalized shifted MINRES (CSR backends; inner solves by CG).
 * \example example_gen_csr.cpp
 * \author Shuntaro Hidaka
 *
 * \details
 * Solves \f$(A + \sigma^{(m)} B)x^{(m)} = b\f$ for multiple complex shifts.
 * Both \c A and \c B are loaded in CSR. The generalized Lanczos steps apply
 * \f$B^{-1}\f$ via a CG inner solve (tolerance can be tuned) and use CSR SpMV
 * for \c A and \c B.
 *
 * **Input**
 *  - `A.mtx` (Hermitian / real-symmetric, CSR after loading)
 *  - `B.mtx` (Hermitian / real-symmetric positive definite, CSR after loading)
 *
 * **Output (per shift m)**
 *  - index, Re(σ), Im(σ), iteration count, algorithmic residual estimate,
 *    and the true 2-norm residual \f$\|(A+\sigma B)x-b\|_2\f$.
 *
 * \par Usage
 * \code
 *   a_gen_csr.out A.mtx B.mtx
 * \endcode
 *
 * \note Inner CG iterations and relative residual are reported inside the code
 *       (variables available if you wish to print them).
 *
 * \see gsi_sminres::generalized::Solver
 * \see gsi_sminres::extras::cg
 * \see gsi_sminres::sparse::SpMV
 * \see gsi_sminres::io::load_mm_csr
 */

#include "gsi_sminres/algorithms/generalized_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include "gsi_sminres/extras/sparse/csr.hpp"
#include "gsi_sminres/extras/sparse/spmv.hpp"
#include "gsi_sminres/extras/io/mm_csr.hpp"
#include "gsi_sminres/extras/algorithms/cg.hpp"
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
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <MTX_file(A)> <MTX_file(B)>" << std::endl;
    return 1;
  }
  std::string Aname = argv[1], Bname = argv[2];
  const auto A = gsi_sminres::io::load_mm_csr_r(Aname, N);
  const auto B = gsi_sminres::io::load_mm_csr_r(Bname, N);
  // Prepare shifts
  std::vector<std::complex<double>> sigma(M);
  for (std::size_t m = 0; m < M; ++m) {
    sigma[m] = std::polar(0.01, 2*std::acos(-1)*(m+0.5)/M);
  }
  // Prepare rhs
  std::vector<double> b(N, 1.0);

  // Prepare variables
  std::vector<std::complex<double>> x(M*N, {0.0, 0.0});
  std::vector<double> w(N, 0.0), u(N, 0.0);
  std::vector<std::size_t> itr(M);
  std::vector<double>      res(M);
  std::size_t inner_iters; double inner_relres;

  auto start = std::chrono::high_resolution_clock::now();

  gsi_sminres::generalized::Solver solver(N, M);
  gsi_sminres::extras::cg_r(B, w, b, N, 1e-13, &inner_iters, &inner_relres);
  solver.initialize_r(x, b, w, sigma, 1e-13);
  for (std::size_t j = 0; j < 10*N; ++j) {
    gsi_sminres::sparse::SpMV_r(A, w, u);
    solver.glanczos_pre_r(u);
    gsi_sminres::extras::cg_r(B, w, u, N, 1e-13, &inner_iters, &inner_relres);
    solver.glanczos_pst_r(w, u);
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
    const auto B_cmplx = gsi_sminres::io::load_mm_csr(Bname, N);
    std::vector<std::complex<double>> b_cmplx(N, {1.0, 0.0});
    std::vector<std::complex<double>> ans(x.begin()+m*N, x.begin()+(m+1)*N);
    std::vector<std::complex<double>> tmp(N), tmpB(N);
    gsi_sminres::sparse::SpMV(A_cmplx, ans, tmp);
    gsi_sminres::sparse::SpMV(B_cmplx, ans, tmpB);
    gsi_sminres::linalg::blas::zaxpy(N, sigma[m], tmpB, 0, tmp, 0);
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
