/**
 * \file shift_invert_shifted_minres.hpp
 * \brief Header file for the Shift-Invert preconditioned shifted MINRES method Solver class
 * \author Shuntaro Hidaka
 * \details This header defines the core 'Solver' class.
 *          The solver implements the Shift-Invert preconditioned Shifted MINRES method,
 *          capable of simultaneously solving a set of shifted linear systems of the form
 *          \f[
 *            (A + \sigma^{(m)} B)x^{(m)} = b, \quad (m = 1, 2, \dots, M)
 *          \f]
 *          where \f$ A \f$ is Hermitian, B is Hermitian positive-definite,
 *          and the shift parameters \f$ \sigma^{(m)} \f$ are complex scalars.
 */

#ifndef GSI_SMINRES_ALGORITHMS_SHIFT_INVERT_SHIFTED_MINRES_HPP
#define GSI_SMINRES_ALGORITHMS_SHIFT_INVERT_SHIFTED_MINRES_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace gsi_sminres {
  /**
   * \namespace gsi_sminres::shift_invert
   * \brief [Solver] Shift-Invert preconditioned shifted MINRES method.
   * \details The `shift_invert` namespace provides the core infrastructure for solving
   *          multiple generalized shifted linear systems using the shift-invert preconditioned shifted MINRES method.
   *
   *          It includes:
   *          - The `Solver` class for orchestrating the shift-invert preconditioned shifted MINRES iteration.
   *          - Numerical routines interfacing with BLAS and LAPACK.
   *          - Utility functions and data structures for managing matrices and vectors.
   */
  namespace shift_invert {
    /**
     * \class Solver
     * \brief Shift-Invert preconditioned shifted MINRES solver class.
     * \details This class solves a set of Generalized shifted linear systems
     *          using the MINRES method and the Shift-Invert preconditioned Lanczos process
     */
    class Solver {
    public:
      /**
       * \brief Constructor.
       * \param[in] matrix_size Matrix size \c N.
       * \param[in] shift_size  Number of shifts \c M.
       */
      Solver(std::size_t matrix_size, std::size_t shift_size);

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
       * \details The caller prepares: \f$ v <- (A + \omega B)^{-1} b, Bv <- B v \f$.
       * \param[out] x     Approximate solutions (row-major, x[m*N+i] size = matrix_size * shift_size).
       * \param[out] v     Pre-processed right-hand side \f$ (A + \omega B)^{-1}b \f$ (size = matrix_size).
       * \param[out] Bv    Vector to which is matrix-vector multiplication is applied (Bv <- B * v).
       * \param[in]  sigma Vector of shift parameters (size = shift_size).
       * \param[in]  omega Shift-Invert parameter.
       * \param[in]  rtol  Convergence tolerance for relative residuals.
       */
      void initialize(std::vector<std::complex<double>>& x,
                      std::vector<std::complex<double>>& v,
                      std::vector<std::complex<double>>& Bv,
                      const std::vector<std::complex<double>>& sigma,
                      const std::complex<double> omega,
                      const double rtol);

      /**
       * \brief Applies the "pre" stage of M-inner-product Lanczos (steps 3.1 -- 3.4).
       * \details The caller prepares: \f$ v <- (A + \omega B)^{-1} B v_j\f$ and provides \f$ Bv_j \f$.
       * \param[in]    Bv Vector to which is matrix-vector multiplication is applied (Bv <- B * v).
       * \param[in,out] v Vector to which is applied the operator \f$ (A + \omega B)^{-1} B \f$ (v <- (A + omega B)^{-1} Bv) (size = N).
       */
      void sislanczos_pre(std::vector<std::complex<double>>& v,
                          const std::vector<std::complex<double>>& Bv) noexcept;

      /**
       * \brief Applies the "post" stage of M-inner-product Lanczos (steps 3.5 -- 3.6)
       * \details The caller prepares: \f$ Bv <- B v \f$.
       * \param[in,out] v  sislanczos_pre の結果をそのまま使う (size = N).
       * \param[in,out] Bv Vector to which is matrix-vector multiplication is applied (Bv <- B * v).
       */
      void sislanczos_pst(std::vector<std::complex<double>>& v,
                          std::vector<std::complex<double>>& Bv) noexcept;

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
      std::complex<double> omega_{};               ///< Shift-Invert parameter \f$ \omega \f$

      // Generalized Lanczos scalars / vectors
      double alpha_{};                   ///< alpha coefficient
      double beta_prev_{}, beta_curr_{}; ///< beta coefficients (previous and current)
      std::vector<std::complex<double>> v_prev_, v_curr_, v_next_; ///< Lanczos basis vectors

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

  }  // namespace shift_invert
}  // namespace gsi_sminres


#endif // GSI_SMINRES_ALGORITHMS_SHIFT_INVERT_SHIFTED_MINRES_HPP
