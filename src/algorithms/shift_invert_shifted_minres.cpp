/**
 * \file shift_invert_shifted_minres.cpp
 * \brief Implementation of the shift-invert preconditioned shifted minres method
 * \author Shuntaro Hidaka
 */

#include "gsi_sminres/algorithms/shift_invert_shifted_minres.hpp"
#include "gsi_sminres/linalg/blas.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
namespace gsi_sminres {
  namespace shift_invert {

    Solver::Solver(std::size_t matrix_size, std::size_t shift_size)
      : iter_(1),
        matrix_size_(matrix_size),
        shift_size_(shift_size),
        r0_norm_(0.0),
        rtol_(1e-12),
        sigma_(shift_size, {0.0, 0.0}),
        omega_({0.0, 0.0}),
        alpha_(0.0),
        beta_prev_(0.0),
        beta_curr_(0.0),
        v_prev_(matrix_size, {0.0, 0.0}),
        v_curr_(matrix_size, {0.0, 0.0}),
        v_next_(matrix_size, {0.0, 0.0}),
        T_prev2_(1, {0.0, 0.0}),
        T_prev_( 1, {0.0, 0.0}),
        T_curr_( 1, {0.0, 0.0}),
        T_next_( 1, {0.0, 0.0}),
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
                            std::vector<std::complex<double>>& v,
                            std::vector<std::complex<double>>& Bv,
                            const std::vector<std::complex<double>>& sigma,
                            const std::complex<double> omega,
                            const double rtol) {
      std::fill(x.begin(), x.end(), std::complex<double>{0.0,0.0});
      r0_norm_ = std::sqrt( std::max(0.0, std::real(linalg::blas::zdotc(matrix_size_, v, 0, Bv, 0))) );
      if (r0_norm_ < rtol) {
        for (std::size_t m = 0; m < shift_size_; ++m) is_conv_[m] = 1u;
        conv_num_ = shift_size_;
        std::fill(h_.begin(), h_.end(), r0_norm_);
        return;
      }
      linalg::blas::zdscal(matrix_size_, 1.0/r0_norm_,  v);
      linalg::blas::zdscal(matrix_size_, 1.0/r0_norm_, Bv);
      linalg::blas::zcopy(matrix_size_, v, 0, v_curr_, 0);
      std::fill(h_.begin(), h_.end(), r0_norm_);
      linalg::blas::zcopy(shift_size_, sigma, 0, sigma_, 0);
      omega_ = omega;
      rtol_ = rtol;
    }

    void Solver::sislanczos_pre(std::vector<std::complex<double>>& v,
                                const std::vector<std::complex<double>>& Bv) noexcept {
      alpha_ = std::real(linalg::blas::zdotc(matrix_size_, Bv, 0, v, 0));
      linalg::blas::zaxpy(matrix_size_, -beta_prev_, v_prev_, 0,  v, 0);
      linalg::blas::zaxpy(matrix_size_, -alpha_,     v_curr_, 0,  v, 0);
    }

    void Solver::sislanczos_pst(std::vector<std::complex<double>>& v,
                                std::vector<std::complex<double>>& Bv) noexcept {
      beta_curr_ = std::sqrt( std::real(linalg::blas::zdotc(matrix_size_, v, 0, Bv, 0)) );
      linalg::blas::zdscal(matrix_size_, 1.0/beta_curr_, v);
      linalg::blas::zdscal(matrix_size_, 1.0/beta_curr_, Bv);
      linalg::blas::zcopy(matrix_size_, v, 0, v_next_, 0);
    }

    [[nodiscard]] bool Solver::update(std::vector<std::complex<double>>& x) noexcept {
      std::swap(p_prev_, p_prev2_);
      std::swap(p_curr_, p_prev_);
      for (std::size_t m = 0; m < shift_size_; ++m) {
        if (is_conv_[m] != 0u) {
          continue;
        }
        T_prev2_[0] = 0.0;
        T_prev_[0]  = (sigma_[m] - omega_) * beta_prev_;
        T_curr_[0]  = 1.0 + (sigma_[m] - omega_) * alpha_;
        T_next_[0]  = (sigma_[m] - omega_) * beta_curr_;
        if (iter_ >= 3) {
          linalg::blas::zrot(1, T_prev2_, 0, T_prev_, 0, Gc_[m][0], Gs_[m][0]);
        }
        if (iter_ >= 2) {
          linalg::blas::zrot(1, T_prev_,  0, T_curr_, 0, Gc_[m][1], Gs_[m][1]);
        }
        linalg::blas::zrotg(T_curr_[0], T_next_[0], Gc_[m][2], Gs_[m][2]);
        //linalg::lapack::zlartg(T_curr_[0], T_next_[0], Gc_[m][2], Gs_[m][2]);
        std::size_t offset = m*matrix_size_;
        linalg::blas::zcopy(matrix_size_, v_curr_, 0, p_curr_, offset);
        linalg::blas::zaxpy(matrix_size_, -T_prev2_[0], p_prev2_, offset, p_curr_, offset);
        linalg::blas::zaxpy(matrix_size_, -T_prev_[0],  p_prev_,  offset, p_curr_, offset);
        linalg::blas::zscal(matrix_size_, 1.0/T_curr_[0], p_curr_, offset);
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
      std::swap(v_curr_, v_prev_);
      std::swap(v_next_, v_curr_);
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

  }  // namespace shift_invert
}  // namespace gsi_sminres
