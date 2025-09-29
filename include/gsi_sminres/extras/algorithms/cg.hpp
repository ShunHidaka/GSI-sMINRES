/**
 * \file cg.hpp
 * \brief Unpreconditioned Conjugate Gradient (CG) solver; compatible with CSR storage.
 * \author Shuntaro Hidaka
 *
 * \details Declares a CG routine that operates on matrices stored in the CSR format as defined in
 *          gsi_sminres/extras/sparse/csr.hpp. Assumes the coefficient matrix is real-symmetric
 *          or Hermitian positive definite. The input vector x is used as an initial
 *          guess and overwritten with the approximate solution.
 */

#ifndef GSI_SMINRES_EXTRAS_ALGORITHMS_CG_HPP
#define GSI_SMINRES_EXTRAS_ALGORITHMS_CG_HPP

#include "gsi_sminres/extras/sparse/csr.hpp"
#include <cstddef>
#include <complex>
#include <vector>

namespace gsi_sminres {
    namespace extras {

      /**
       * \brief Solve \f$ Ax=b \f$ using the (unpreconditioned) Conjugate Gradient (CG) method.
       *
       * \details Iteration stops when the relative residual \f$ \|r_k\|_2 / \|b\|_2 \le \mathrm{rtol} \f$
       *          or when \p max_iter iterations are reached.
       *          On exit, \p x always contains the last iterate (useful even if not converged).
       *
       * \param[in]     A        Coefficient matrix (real-symmetric or Hermitian positive definite),
       *                         stored in CSR format as defined in gsi_sminres/extras/sparse/csr.hpp.
       * \param[in,out] x        Solution vector; on input an initial guess, on output the approximate solution.
       * \param[in]     b        Right-hand side vector.
       * \param[in]     max_iter Maximum number of iterations.
       * \param[in]     rtol     Relative residual tolerance.
       * \param[out]    iters    Number of iterations performed (set even if the method does not converge).
       * \param[out]    relres   Final relative residual \f$ \|r\|_2 / \|b\|_2 \f$ at exit.
       *
       * \returns true if the tolerance is satisfied within \p max_iter; false otherwise.
       *
       * \pre A is stored in CSR format as defined in gsi_sminres/extras/sparse/csr.hpp.
       * \pre A is square and SPD/HPD.
       */
      bool cg(const sparse::CSRMatrix&                 A,
              std::vector<std::complex<double>>&       x,
              const std::vector<std::complex<double>>& b,
              std::size_t                              max_iter,
              double                                   rtol,
              std::size_t*                             iters = nullptr,
              double*                                  relres = nullptr);

  }  // namespace extras
}  // namespace gsi_sminres


#endif // GSI_SMINRES_EXTRAS_ALGORITHMS_CG_HPP
