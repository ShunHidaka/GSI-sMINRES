/**
 * \file standard_shifted_minres.cpp
 * \brief Implementation of the standard shifted minres method
 * \author Shuntaro Hidaka
 */

#include "gsi_sminres/algorithms/standard_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include <algorithm>
#include <cmath>

namespace gsi_sminres {
  namespace standard {

    Solver::Solver(std::size_t matrix_size, std::size_t shift_size)
      : iter_(1),
        matrix_size_(matrix_size),
        shift_size_(shift_size),
        r0_norm_(0.0),
        rtol_(1e-12),
        sigma_(shift_size, {0.0, 0.0}),
        alpha_(0.0),
        beta_prev_(0.0),
        beta_curr_(0.0),
        v_prev_(matrix_size, {0.0, 0.0}),
        v_curr_(matrix_size, {0.0, 0.0}),
        v_next_(matrix_size, {0.0, 0.0}),
        Gc_(shift_size, std::array<double,3>{0.0, 0.0, 0.0}),
        Gs_(shift_size, std::array<std::complex<double>,3>{{{0.0,0.0}, {0.0,0.0}, {0.0,0.0}}}),
        p_prev2_(shift_size*matrix_size, {0.0, 0.0}),
        p_prev_( shift_size*matrix_size, {0.0, 0.0}),
        p_curr_( shift_size*matrix_size, {0.0, 0.0}),
        f_(shift_size, {1.0, 0.0}),
        h_(shift_size, 1.0),
        conv_num_(0),
        is_conv_(shift_size, 0u),
        conv_iter_(shift_size, 0) {}

    void Solver::initialize(std::vector<std::complex<double>>& x,
                            const std::vector<std::complex<double>>& b,
                            std::vector<std::complex<double>>& v,
                            const std::vector<std::complex<double>>& sigma,
                            const double rtol) {
      real_lanczos_mode_ = false; // Complex-valued Lanczos mode

      std::fill(x.begin(), x.end(), std::complex<double>{0.0,0.0});
      r0_norm_ = linalg::blas::dznrm2(matrix_size_, b);
      if (r0_norm_ < rtol) {
        for (std::size_t m = 0; m < shift_size_; ++m) is_conv_[m] = 1u;
        conv_num_ = shift_size_;
        std::fill(h_.begin(), h_.end(), r0_norm_);
        return;
      }
      linalg::blas::zcopy(matrix_size_, b, 0, v_curr_, 0);
      linalg::blas::zdscal(matrix_size_, 1.0/r0_norm_, v_curr_);
      linalg::blas::zcopy(matrix_size_, v_curr_, 0, v, 0);
      beta_prev_ = 0.0;
      iter_      = 1;
      std::fill(h_.begin(), h_.end(), r0_norm_);
      linalg::blas::zcopy(shift_size_, sigma, 0, sigma_, 0);
      rtol_ = rtol;
    }

    void Solver::initialize_r(std::vector<std::complex<double>>& x,
                              const std::vector<double>& b,
                              std::vector<double>& v,
                              const std::vector<std::complex<double>>& sigma,
                              const double rtol) {
      real_lanczos_mode_ = true; // Real-valued Lanczos mode

      std::fill(x.begin(), x.end(), std::complex<double>{0.0,0.0});
      r0_norm_ = linalg::blas::dnrm2(matrix_size_, b);
      if (r0_norm_ < rtol) {
        for (std::size_t m = 0; m < shift_size_; ++m) is_conv_[m] = 1u;
        conv_num_ = shift_size_;
        std::fill(h_.begin(), h_.end(), r0_norm_);
        return;
      }
      v_prev_r_.assign(matrix_size_, 0.0);
      v_curr_r_.assign(matrix_size_, 0.0);
      v_next_r_.assign(matrix_size_, 0.0);
      linalg::blas::dcopy(matrix_size_, b, 0, v_curr_r_, 0);
      linalg::blas::dscal(matrix_size_, 1.0/r0_norm_, v_curr_r_);
      linalg::blas::dcopy(matrix_size_, v_curr_r_, 0, v, 0);
      for (std::size_t i = 0; i < matrix_size_; ++i) {
        v_curr_[i] = std::complex<double>(v_curr_r_[i], 0.0);
      }
      beta_prev_ = 0.0;
      iter_      = 1;
      std::fill(h_.begin(), h_.end(), r0_norm_);
      linalg::blas::zcopy(shift_size_, sigma, 0, sigma_, 0);
      rtol_ = rtol;
    }

    void Solver::lanczos(std::vector<std::complex<double>>& v,
                         const std::vector<std::complex<double>>& Av) noexcept {
      linalg::blas::zcopy(matrix_size_, Av, 0, v, 0);
      alpha_ = std::real(linalg::blas::zdotc(matrix_size_, v_curr_, 0, v, 0));
      linalg::blas::zaxpy(matrix_size_, -alpha_,     v_curr_, 0, v, 0);
      linalg::blas::zaxpy(matrix_size_, -beta_prev_, v_prev_, 0, v, 0);
      beta_curr_ = linalg::blas::dznrm2(matrix_size_, v);
      linalg::blas::zdscal(matrix_size_, 1.0/beta_curr_, v);
      linalg::blas::zcopy(matrix_size_, v, 0, v_next_, 0);
    }
    void Solver::lanczos_r(std::vector<double>& v,
                         const std::vector<double>& Av) noexcept {
      linalg::blas::dcopy(matrix_size_, Av, 0, v, 0);
      alpha_ = linalg::blas::ddot(matrix_size_, v_curr_r_, 0, v, 0);
      linalg::blas::daxpy(matrix_size_, -alpha_,     v_curr_r_, 0, v, 0);
      linalg::blas::daxpy(matrix_size_, -beta_prev_, v_prev_r_, 0, v, 0);
      beta_curr_ = linalg::blas::dnrm2(matrix_size_, v);
      linalg::blas::dscal(matrix_size_, 1.0/beta_curr_, v);
      linalg::blas::dcopy(matrix_size_, v, 0, v_next_r_, 0);
    }


    [[nodiscard]] bool Solver::update(std::vector<std::complex<double>>& x) noexcept {
      std::swap(p_prev_, p_prev2_);
      std::swap(p_curr_, p_prev_);
#pragma omp parallel for
      for (std::size_t m = 0; m < shift_size_; ++m) {
        if (is_conv_[m] != 0u) {
          continue;
        }
        std::complex<double> T_prev2_ = 0.0;
        std::complex<double> T_prev_  = beta_prev_;
        std::complex<double> T_curr_  = alpha_ + sigma_[m];
        std::complex<double> T_next_  = beta_curr_;
        if (iter_ >= 3) {
          linalg::blas::apply_givens(Gc_[m][0], Gs_[m][0], T_prev2_, T_prev_);
        }
        if (iter_ >= 2) {
          linalg::blas::apply_givens(Gc_[m][1], Gs_[m][1], T_prev_, T_curr_);
        }
        linalg::blas::zrotg(T_curr_, T_next_, Gc_[m][2], Gs_[m][2]);
        //linalg::lapack::zlartg(T_curr_, T_next_, Gc_[m][2], Gs_[m][2]);
        std::size_t offset = m*matrix_size_;
        linalg::blas::zcopy(matrix_size_, v_curr_, 0, p_curr_, offset);
        linalg::blas::zaxpy(matrix_size_, -T_prev2_, p_prev2_, offset, p_curr_, offset);
        linalg::blas::zaxpy(matrix_size_, -T_prev_,  p_prev_,  offset, p_curr_, offset);
        linalg::blas::zscal(matrix_size_, 1.0/T_curr_, p_curr_, offset);
        linalg::blas::zaxpy(matrix_size_, r0_norm_*Gc_[m][2]*f_[m], p_curr_, offset, x, offset);
        f_[m] = -std::conj(Gs_[m][2]) * f_[m];
        h_[m] = std::abs(-std::conj(Gs_[m][2])) * h_[m];
        if (h_[m]/r0_norm_ < rtol_) {
          conv_num_++;
          is_conv_[m] = 1u;
          conv_iter_[m] = iter_;
          continue;
        }
        Gc_[m][0] = Gc_[m][1]; Gc_[m][1] = Gc_[m][2];
        Gs_[m][0] = Gs_[m][1]; Gs_[m][1] = Gs_[m][2];
      }
      beta_prev_ = beta_curr_;
      if (real_lanczos_mode_) {
        std::swap(v_curr_r_, v_prev_r_);
        std::swap(v_next_r_, v_curr_r_);
        for (std::size_t i = 0; i < matrix_size_; ++i) {
          v_curr_[i] = std::complex<double>(v_curr_r_[i], 0.0);
        }
      } else {
        std::swap(v_curr_, v_prev_);
        std::swap(v_next_, v_curr_);
      }
      iter_++;
      if (conv_num_ >= shift_size_) {
        return true;
      }
      return false;
    }

    void Solver::finalize(std::vector<std::size_t>& conv_itr,
                          std::vector<double>&      conv_res) const {
      // 当初はメモリの解放などを行う予定だったが
      // (動的な確保をおこなっていないため)不要なので収束までの反復回数と残差のノルムを返す関数とする
      //conv_itr = conv_iter_;
      for (std::size_t m = 0; m < shift_size_; ++m)
        conv_itr[m] = (is_conv_[m] != 0u) ? conv_iter_[m] : iter_;
      conv_res = h_;
    }

    void Solver::get_alg_residual(std::vector<double>& res) const noexcept {
      res = h_;
    }

  }  // namespace standard
}  // namespace gsi_sminres
