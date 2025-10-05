/**
 * \file example_sis_csr.cpp
 * \brief Demo: Shift–Invert preconditioned shifted MINRES with CSR backends.
 * \example example_sis_csr.cpp
 * \author Shuntaro Hidaka
 *
 * \details
 * Solves a batch of shifted linear systems
 * \f$(A + \sigma^{(m)} B) x^{(m)} = b \quad (m=0,\dots,M-1)\f$
 * using the Shift–Invert preconditioned shifted MINRES solver.
 * Matrices \c A and \c B are loaded from Matrix Market files and stored in CSR.
 *
 * The shift–invert preconditioner applies \f$(A + \omega B)^{-1}\f$ to vectors;
 * this example sets \f$\omega=0\f$ and uses a helper MINRES routine on the
 * matrix pencil \f$(A, B)\f$ in CSR to realize the solve.
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
 *   a_sis_csr.out A.mtx B.mtx
 * \endcode
 *
 * \note This is a minimal end-to-end example intended for correctness checks,
 *       not a performance benchmark. The solution array \c x is laid out as
 *       M contiguous length-N blocks (one block per shift).
 *
 * \see gsi_sminres::shift_invert::Solver
 * \see gsi_sminres::io::load_mm_csr
 * \see gsi_sminres::sparse::SpMV
 * \see gsi_sminres::extras::minres_pencil
 */

#include "gsi_sminres/algorithms/shift_invert_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include "gsi_sminres/extras/sparse/csr.hpp"
#include "gsi_sminres/extras/sparse/spmv.hpp"
#include "gsi_sminres/extras/io/mm_csr.hpp"
#include "gsi_sminres/extras/algorithms/minres_pencil.hpp"
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
  const auto A = gsi_sminres::io::load_mm_csr(Aname, N);
  const auto B = gsi_sminres::io::load_mm_csr(Bname, N);
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
  std::size_t inner_iters; double inner_relres;

  // Constructing shift-invert preconditioner
  const std::complex<double> omega = 0.0;

  gsi_sminres::shift_invert::Solver solver(N, M);
  gsi_sminres::extras::minres_pencil(A, omega, B, v, b, N, 1e-13, &inner_iters, &inner_relres);
  gsi_sminres::sparse::SpMV(B, v, Bv);
  solver.initialize(x, v, Bv, sigma, omega, 1e-12);
  for (std::size_t j = 0; j < N; ++j) {
    gsi_sminres::extras::minres_pencil(A, omega, B, v, Bv, N, 1e-13, &inner_iters, &inner_relres);
    solver.sislanczos_pre(v, Bv);
    gsi_sminres::sparse::SpMV(B, v, Bv);
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
    std::vector<std::complex<double>> tmp(N), tmpB(N);
    gsi_sminres::sparse::SpMV(A, ans, tmp);
    gsi_sminres::sparse::SpMV(B, ans, tmpB);
    gsi_sminres::linalg::blas::zaxpy(N, sigma[m], tmpB, 0, tmp, 0);
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
