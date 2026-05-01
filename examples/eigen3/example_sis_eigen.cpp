
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "eigen_csr_view.hpp"
#include "gsi_sminres/algorithms/shift_invert_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include "gsi_sminres/extras/sparse/csr.hpp"
#include "gsi_sminres/extras/io/mm_csr.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <vector>
#include <chrono>
#include <string>

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
  const double omega = 0.01;
  std::vector<std::complex<double>> sigma(M);
  for (std::size_t m = 0; m < M; ++m) {
    sigma[m] = omega + std::polar(0.1, 2*std::acos(-1)*(m+0.5)/M);
  }
  // Prepare rhs
  std::vector<std::complex<double>> b(N, {1.0, 0.0});

  // Prepare variables
  std::vector<std::complex<double>> x(M*N, {0.0, 0.0});
  std::vector<std::complex<double>> v(N, {0.0, 0.0}), Bv(N, {0.0, 0.0});
  std::vector<std::size_t> itr(M);
  std::vector<double>      res(M);
  std::size_t inner_iters; double inner_relres;

  // Prepare Eigen matrix view
  auto A_view = eigen_bridge::make_eigen_csr_view(A); auto A_eigen_map = A_view.map();
  auto B_view = eigen_bridge::make_eigen_csr_view(B); auto B_eigen_map = B_view.map();
  // Build pencil matrix: P = A + omega B
  eigen_bridge::SpMatRow P = (A_eigen_map + omega*B_eigen_map).eval();
  P.makeCompressed();
  // Prepare Eigen vector view
  Eigen::Map<const Eigen::VectorXcd> bv(b.data(), static_cast<Eigen::Index>(N));
  Eigen::Map<Eigen::VectorXcd> vv(v.data(), static_cast<Eigen::Index>(N));
  Eigen::Map<Eigen::VectorXcd> Bvv(Bv.data(), static_cast<Eigen::Index>(N));
  // Prepare Eigen BiCGStab solver
  Eigen::BiCGSTAB<eigen_bridge::SpMatRow,
                  Eigen::IdentityPreconditioner
                  //Eigen::IncompleteLUT<eigen_bridge::Scalar>
                  > bicgs;
  bicgs.setMaxIterations(static_cast<Eigen::Index>(10*N));
  bicgs.setTolerance(1e-12);
  //bicgs.preconditioner().setFillfactor(20);
  //bicgs.preconditioner().setDroptol(1e-3);
  bicgs.compute(P);
  if (bicgs.info() != Eigen::Success) {
    std::cerr << "Eigen BiCGSTAB compute(P) failed" << std::endl;
    return 1;
  }

  auto start = std::chrono::high_resolution_clock::now();

  gsi_sminres::shift_invert::Solver solver(N, M);
  vv = bicgs.solve(bv);
  inner_iters = static_cast<std::size_t>(bicgs.iterations()); inner_relres = bicgs.error();
  std::cout << "# " << inner_iters << ", " << inner_relres << std::endl;
  Bvv = B_eigen_map * vv;
  solver.initialize(x, v, Bv, sigma, omega, 1e-12);
  for (std::size_t j = 0; j < N; ++j) {
    vv = bicgs.solveWithGuess(Bvv, vv);
    inner_iters = static_cast<std::size_t>(bicgs.iterations()); inner_relres = bicgs.error();
    std::cerr << "# " << j << ": " << inner_iters << ", " << inner_relres << std::endl;
    solver.sislanczos_pre(v, Bv);
    Bvv = B_eigen_map * vv;
    solver.sislanczos_pst(v, Bv);
    if (solver.update(x)) {
      break;
    }
    solver.get_alg_residual(res);
  }
  solver.finalize(itr, res);

  auto end = std::chrono::high_resolution_clock::now();

  // Output results
  double sec = std::chrono::duration<double>(end - start).count();
  std::cout << "# sis-MINRES method (ILU-BiCGSTAB)" << std::endl;
  std::cout << "# A = " << Aname << "\n"
            << "# B = " << Bname << std::endl;
  std::cout << "# time = " << sec << " s" << std::endl;
  for (std::size_t m = 0; m < M; ++m) {
    std::vector<std::complex<double>> ans(x.begin()+m*N, x.begin()+(m+1)*N);
    std::vector<std::complex<double>> tmp(N), tmpB(N);
    Eigen::Map<const Eigen::VectorXcd> ansv(ans.data(), static_cast<Eigen::Index>(N));
    Eigen::Map<Eigen::VectorXcd> tmpv(tmp.data(), static_cast<Eigen::Index>(N));
    Eigen::Map<Eigen::VectorXcd> tmpBv(tmpB.data(), static_cast<Eigen::Index>(N));
    tmpv  = A_eigen_map * ansv;
    tmpBv = B_eigen_map * ansv;
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
