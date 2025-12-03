/**
 * \file minres_pencil.hpp
 * \brief Unpreconditioned MINRES solver; compatible with CSR storage.
 * \author Shuntaro Hidaka
 *
 * \details Declares a MINRES routine that operates on matrices stored in the CSR format as defined in
 *          gsi_sminres/extras/sparse/csr.hpp. Assumes the coefficient matrix is real-symmetric or Hermitian.
 *          The input vector x is used as an initial guess and overwritten with the approximate solution.
 */

#ifndef GSI_SMINRES_EXTRAS_ALGORITHMS_MINRES_PENCIL_HPP
#define GSI_SMINRES_EXTRAS_ALGORITHMS_MINRES_PENCIL_HPP

#include "gsi_sminres/extras/sparse/csr.hpp"
#include <cstddef>
#include <complex>
#include <vector>

namespace gsi_sminres {
    namespace extras {

      /**
       * \brief Solve \f$ (A + \omega B)x=b \f$ using the (unpreconditioned) MINRES method.
       *
       * \details Iteration stops when the relative residual \f$ \|r_k\|_2 / \|b\|_2 \le \mathrm{rtol} \f$
       *          or when \p max_iter iterations are reached.
       *          On exit, \p x always contains the last iterate (useful even if not converged).
       *
       * \param[in]     A        Coefficient matrix \f$ A \f$ (real-symmetric or Hermitian),
       *                         stored in CSR format as defined in gsi_sminres/extras/sparse/csr.hpp.
       * \param[in]     omega    Shift parameter \f$ \omega \f$ (accepted as complex for BLAS convenience, but MINRES requires
       *                         \f$ \omega \in \mathbb{R} \f$ so that \f$ A + \omega B \f$ is real-symmetric or Hermitian).
       * \param[in]     B        Coefficient matrix \f$ B \f$ (real-symmetric or Hermitian),
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
       * \pre A and B are stored in CSR format as defined in gsi_sminres/extras/sparse/csr.hpp.
       * \pre A and B are square and real-symmetric or Hermitian.
       * \pre \f$ \omega \in \mathbb{R} \f$.
       */
      bool minres_pencil(const sparse::CSRMatrix&                 A,
                         const std::complex<double>               omega,
                         const sparse::CSRMatrix&                 B,
                         std::vector<std::complex<double>>&       x,
                         const std::vector<std::complex<double>>& b,
                         std::size_t                              max_iter,
                         double                                   rtol,
                         std::size_t*                             iters = nullptr,
                         double*                                  relres = nullptr);

      bool minres_pencil_r(const sparse::CSRMatrix_r& A,
                           const double               omega,
                           const sparse::CSRMatrix_r& B,
                           std::vector<double>&       x,
                           const std::vector<double>& b,
                           std::size_t                max_iter,
                           double                     rtol,
                           std::size_t*               iters = nullptr,
                           double*                    relres = nullptr);

    }  // namespace extras
}  // namespace gsi_sminres


#endif // GSI_SMINRES_EXTRAS_ALGORITHMS_MINRES_PENCIL_HPP
