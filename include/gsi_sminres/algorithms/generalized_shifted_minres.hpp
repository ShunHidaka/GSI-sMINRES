/**
 * \file generalized_shifted_minres.hpp
 * \brief Header file for the Generalized shifted MINRES method Solver class
 * \author Shuntaro Hidaka
 * \details This header defines the core 'Solver' class.
 *          The solver implements the Generalized Shifted MINRES method,
 *          capable of simultaneously solving a set of shifted linear systems of the form
 *          \f[
 *            (A + \sigma^{(m)} B)x^{(m)} = b, \quad (m = 1, 2, \dots, M)
 *          \f]
 *          where A is Hermitian, B is Hermitian positive-definite,
 *          and the shift parameters \f$ \sigma^{(m)} \f$ are complex scalars.
 *
 */

#ifndef GSI_SMINRES_ALGORITHMS_GENERALIZED_SHIFTED_MINRES_HPP
#define GSI_SMINRES_ALGORITHMS_GENERALIZED_SHIFTED_MINRES_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace gsi_sminres {
  /**
   * \namespace gsi_sminres::generalized
   * \brief [Solver] Generalized shifted MINRES method.
   * \details The `generalized` namespace provides the core infrastructure for solving
   *          multiple generalized shifted linear systems using the generalized shifted MINRES method.
   *
   *          It includes:
   *          - The `Solver` class for orchestrating the generalized shifted MINRES iteration.
   *          - Numerical routines interfacing with BLAS and LAPACK.
   *          - Utility functions and data structures for managing matrices and vectors.
   */
  namespace generalized {
    /**
     * \class Solver
     * \brief Generalized shifted MINRES solver class.
     * \details This class solves a set of generalized shifted linear systems
     *          using the MINRES method and the generalized Lanczos process.
     */
    class Solver {
    public:
      /**
       * \brief Constructor.
       * \param[in] matrix_size Matrix size \c N.
       * \param[in] shift_size  Number of shifts \c M.
       */
      explicit Solver(std::size_t matrix_size, std::size_t shift_size);

      /**
       * \brief Destructor.
       * \details Default destructor. No manual cleanup required.
       */
      ~Solver() noexcept = default;

      // No copying; enforce unique ownership.
      Solver(const Solver&)                = delete;
      Solver& operator=(const Solver&)     = delete;
      // Enable noexcept moves for efficient transfer.
      Solver(Solver&&) noexcept            = default;
      Solver& operator=(Solver&&) noexcept = default;

      /**
       * \brief Initialize the solver with input data and prepare for iteration.
       * \param[out]    x     Approximate solutions (row-major, x[m*N+i] size = matrix_size * shift_size).
       * \param[in]     b     Right-hand side vector (size = matrix_size).
       * \param[in,out] w     Pre-processed right-hand side \f$ B^{-1}b \f$ (size = matrix_size).
       * \param[in]     sigma Vector of shift parameters (size = shift_size).
       * \param[in]     rtol  Convergence tolerance for relative residuals.
       */
      void initialize(std::vector<std::complex<double>>& x,
                      const std::vector<std::complex<double>>& b,
                      std::vector<std::complex<double>>& w,
                      const std::vector<std::complex<double>>& sigma,
                      const double rtol);

      /**
       * \brief Pre-step of the generalized Lanczos process.
       * \details The caller computes \f$ u \leftarrow A w \f$ and passes it here.
       * \param[in,out] u Vector containing \f$ u=Aw\f$ (size = N).
       */
      void glanczos_pre(std::vector<std::complex<double>>& u) noexcept;

      /**
       * \brief Post-step of the generalized Lanczos process.
       * \details The caller computes \f$ w \leftarrow B^{-1} u \f$ and passes it back.
       * \param[in,out] w Vector containing \f$ w = B^{-1}u \f$ (size = N).
       * \param[in,out] u Vector which used in `glanczos_pre()` (size=N).
       */
      void glanczos_pst(std::vector<std::complex<double>>& w,
                        std::vector<std::complex<double>>& u) noexcept;

      /**
       * \brief Update the approximate solutions and check convergence.
       * \param[in,out] x Solution vectors to be updated (size = matrix_size * shift_size)
       * \return true if all systems have converged, false otherwise.
       */
      [[nodiscard]] bool update(std::vector<std::complex<double>>& x) noexcept;

      /**
       * \brief Retrieve converged iteration and converged residual norm.
       * \details This function does not finalize or delete the solver instance.
       *          It only retrieves the number of iterations and the residual norm in the Algorithm
       *          at convergence for each shifted system.
       * \param[out] conv_itr Number of iterations for each shift (size = shift_size).
       * \param[out] conv_res Final residual norms in the algorithm for each shift (size = shift_size).
       */
      void finalize(std::vector<std::size_t>& conv_itr, std::vector<double>& conv_res) const;

      /**
       * \brief Retrieve current absolute residual norms in the algorithm.
       * \param[out] res Residual norms in the algorithm for each shift (size = shift_size).
       */
      void get_alg_residual(std::vector<double>& res) const noexcept;

    private:
      // Basic parameters
      std::size_t iter_{};        ///< Current iterations (1-based)
      std::size_t matrix_size_{}; ///< Matrix size \f$ N \f$
      std::size_t shift_size_{};  ///< Number of shifts \f$ M \f$

      // Control / shifts
      double r0_norm_{};                          ///< Norm of initial residual \f$ ||r_0|| \f$
      double rtol_{};                             ///< Relative residual convergence tolerance
      std::vector<std::complex<double>> sigma_{}; ///< Shift values \f$ \sigma^{(m)} \f$

      // Generalized Lanczos scalars / vectors
      double alpha_{};                   ///< alpha coefficient
      double beta_prev_{}, beta_curr_{}; ///< beta coefficients (previous and current)
      std::vector<std::complex<double>> w_prev_, w_curr_, w_next_; ///< Lanczos basis vectors
      std::vector<std::complex<double>> u_prev_, u_curr_, u_next_; ///< Auxiliary vectors

      // tridiagonal / Givens data for MINRES updates
      /**
       * @brief Elements of the tridiagonal matrix by Lanczos process
       *        These vectors store the column-wise elements of a tridiagonal matrix,
       *        used during the Lanczos process. Although each vector contains only one
       *        element in practice, they are wrapped in 'std::vector' to be compatible
       *        with my BLAS routine wrapper that require vector inputs.
       */
      std::vector<std::complex<double>> T_prev2_, T_prev_, T_curr_, T_next_;
      std::vector<std::array<double,3>>               Gc_; ///< Givens rotation matrix elements "c"
      std::vector<std::array<std::complex<double>,3>> Gs_; ///< Givens rotation matrix elements "s"

      // auxiliary buffers
      std::vector<std::complex<double>> p_prev2_, p_prev_, p_curr_; ///< Auxiliary vectors (shift*matrix)
      std::vector<std::complex<double>> f_;                         ///< Auxiliary variables
      std::vector<double> h_;                                       ///< Residual norms in the algorithm

      // Convergence bookkeeping
      std::size_t conv_num_{};             ///< Number of systems that have converged
      std::vector<unsigned char> is_conv_; ///< Flags indicating convergence for each system
      std::vector<std::size_t> conv_iter_; ///< Iteration at which each system converged
    };

  }  // namespace generalized
}  // namespace gsi_sminres

#endif // GSI_SMINRES_ALGORITHMS_GENERALIZED_SHIFTED_MINRES_HPP
