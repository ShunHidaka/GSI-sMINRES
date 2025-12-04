/**
 * \file generalized_shifted_minres.cpp
 * \brief Implementation of the generalized shifted minres method
 * \author Shuntaro Hidaka
 */

#include "gsi_sminres/algorithms/generalized_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include <algorithm>
#include <cmath>

namespace gsi_sminres {
  namespace generalized {

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
        w_prev_(matrix_size, {0.0, 0.0}),
        w_curr_(matrix_size, {0.0, 0.0}),
        w_next_(matrix_size, {0.0, 0.0}),
        u_prev_(matrix_size, {0.0, 0.0}),
        u_curr_(matrix_size, {0.0, 0.0}),
        u_next_(matrix_size, {0.0, 0.0}),
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
                            std::vector<std::complex<double>>& w,
                            const std::vector<std::complex<double>>& sigma,
                            const double rtol) {
      real_lanczos_mode_ = false;

      std::fill(x.begin(), x.end(), std::complex<double>{0.0,0.0});
      r0_norm_ = std::sqrt( std::max(0.0, std::real(linalg::blas::zdotc(matrix_size_, b, 0, w, 0))) );
      if (r0_norm_ < rtol) {
        for (std::size_t m = 0; m < shift_size_; ++m) is_conv_[m] = 1u;
        conv_num_ = shift_size_;
        std::fill(h_.begin(), h_.end(), r0_norm_);
        return;
      }
      linalg::blas::zcopy(matrix_size_, w, 0, w_curr_, 0);
      linalg::blas::zcopy(matrix_size_, b, 0, u_curr_, 0);
      linalg::blas::zdscal(matrix_size_, 1.0/r0_norm_, w_curr_);
      linalg::blas::zdscal(matrix_size_, 1.0/r0_norm_, u_curr_);
      linalg::blas::zcopy(matrix_size_, w_curr_, 0, w, 0);
      beta_prev_ = 0.0;
      iter_      = 1;
      std::fill(h_.begin(), h_.end(), r0_norm_);
      linalg::blas::zcopy(shift_size_, sigma, 0, sigma_, 0);
      rtol_ = rtol;
    }
    void Solver::initialize_r(std::vector<std::complex<double>>& x,
                            const std::vector<double>& b,
                            std::vector<double>& w,
                            const std::vector<std::complex<double>>& sigma,
                            const double rtol) {
      real_lanczos_mode_ = true;

      std::fill(x.begin(), x.end(), std::complex<double>{0.0,0.0});
      r0_norm_ = std::sqrt( std::max(0.0, std::real(linalg::blas::ddot(matrix_size_, b, 0, w, 0))) );
      if (r0_norm_ < rtol) {
        for (std::size_t m = 0; m < shift_size_; ++m) is_conv_[m] = 1u;
        conv_num_ = shift_size_;
        std::fill(h_.begin(), h_.end(), r0_norm_);
        return;
      }
      w_prev_r_.assign(matrix_size_, 0.0);
      w_curr_r_.assign(matrix_size_, 0.0);
      w_next_r_.assign(matrix_size_, 0.0);
      u_prev_r_.assign(matrix_size_, 0.0);
      u_curr_r_.assign(matrix_size_, 0.0);
      u_next_r_.assign(matrix_size_, 0.0);
      linalg::blas::dcopy(matrix_size_, w, 0, w_curr_r_, 0);
      linalg::blas::dscal(matrix_size_, 1.0/r0_norm_, w_curr_r_);
      linalg::blas::dcopy(matrix_size_, b, 0, u_curr_r_, 0);
      linalg::blas::dscal(matrix_size_, 1.0/r0_norm_, u_curr_r_);
      linalg::blas::dcopy(matrix_size_, w_curr_r_, 0, w, 0);
      for (std::size_t i = 0; i < matrix_size_; ++i) {
        w_curr_[i] = std::complex<double>(w_curr_r_[i], 0.0);
      }
      beta_prev_ = 0.0;
      iter_      = 1;
      std::fill(h_.begin(), h_.end(), r0_norm_);
      linalg::blas::zcopy(shift_size_, sigma, 0, sigma_, 0);
      rtol_ = rtol;
    }

    void Solver::glanczos_pre(std::vector<std::complex<double>>& u) noexcept {
      alpha_ = std::real(linalg::blas::zdotc(matrix_size_, w_curr_, 0, u, 0));
      linalg::blas::zaxpy(matrix_size_, -alpha_,     u_curr_, 0, u, 0);
      linalg::blas::zaxpy(matrix_size_, -beta_prev_, u_prev_, 0, u, 0);
    }
    void Solver::glanczos_pre_r(std::vector<double>& u) noexcept {
      alpha_ = linalg::blas::ddot(matrix_size_, w_curr_r_, 0, u, 0);
      linalg::blas::daxpy(matrix_size_, -alpha_,     u_curr_r_, 0, u, 0);
      linalg::blas::daxpy(matrix_size_, -beta_prev_, u_prev_r_, 0, u, 0);
    }

    void Solver::glanczos_pst(std::vector<std::complex<double>>& w,
                              std::vector<std::complex<double>>& u) noexcept {
      beta_curr_ = std::sqrt(std::real(linalg::blas::zdotc(matrix_size_, u, 0, w, 0)));
      linalg::blas::zdscal(matrix_size_, 1.0/beta_curr_, w);
      linalg::blas::zdscal(matrix_size_, 1.0/beta_curr_, u);
      linalg::blas::zcopy(matrix_size_, w, 0, w_next_, 0);
      linalg::blas::zcopy(matrix_size_, u, 0, u_next_, 0);
    }
    void Solver::glanczos_pst_r(std::vector<double>& w,
                                std::vector<double>& u) noexcept {
      beta_curr_ = std::sqrt(linalg::blas::ddot(matrix_size_, u, 0, w, 0));
      linalg::blas::dscal(matrix_size_, 1.0/beta_curr_, w);
      linalg::blas::dscal(matrix_size_, 1.0/beta_curr_, u);
      linalg::blas::dcopy(matrix_size_, w, 0, w_next_r_, 0);
      linalg::blas::dcopy(matrix_size_, u, 0, u_next_r_, 0);
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
        linalg::blas::zcopy(matrix_size_, w_curr_, 0, p_curr_, offset);
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
        std::swap(w_curr_r_, w_prev_r_); std::swap(w_next_r_, w_curr_r_);
        std::swap(u_curr_r_, u_prev_r_); std::swap(u_next_r_, u_curr_r_);
        for (std::size_t i = 0; i < matrix_size_; ++i) {
          w_curr_[i] = std::complex<double>(w_curr_r_[i], 0.0);
        }
      } else {
        std::swap(w_curr_, w_prev_); std::swap(w_next_, w_curr_);
        std::swap(u_curr_, u_prev_); std::swap(u_next_, u_curr_);
      }
      iter_++;
      if (conv_num_ >= shift_size_) {
        return true;
      }
      return false;
    }

    void Solver::finalize(std::vector<std::size_t>& conv_itr,
                          std::vector<double>&      conv_res) const{
      //conv_itr = conv_iter_;
      for (std::size_t m = 0; m < shift_size_; ++m)
        conv_itr[m] = (is_conv_[m] != 0u) ? conv_iter_[m] : iter_;
      conv_res = h_;
    }

    void Solver::get_alg_residual(std::vector<double>& res) const noexcept{
      res = h_;
    }

  }  // namespace generalized
}  // namespace gsi_sminres
