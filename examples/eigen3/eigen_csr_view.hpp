#pragma once

#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include "gsi_sminres/extras/sparse/csr.hpp"

namespace eigen_bridge {
  
  using Scalar   = std::complex<double>;
  using SpMatRow = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, int>;

  struct EigenCSRView {
    Eigen::Index n = 0;
    Eigen::Index nnz = 0;
    std::vector<int> row_ptr_i;
    std::vector<int> col_idx_i;
    const Scalar* values_ptr = nullptr;

    Eigen::Map<const SpMatRow> map() const {
      return Eigen::Map<const SpMatRow>(n, n, nnz,
                                        row_ptr_i.data(),
                                        col_idx_i.data(),
                                        values_ptr,
                                        nullptr);
    }
  };

  inline EigenCSRView make_eigen_csr_view(const gsi_sminres::sparse::CSRMatrix& A) {
    EigenCSRView out;
    out.n = static_cast<Eigen::Index>(A.n);
    out.nnz = static_cast<Eigen::Index>(A.values.size());
    out.values_ptr = A.values.data();

    out.row_ptr_i.reserve(A.row_ptr.size());
    out.col_idx_i.reserve(A.col_idx.size());

    for (auto v : A.row_ptr) out.row_ptr_i.push_back(static_cast<int>(v));
    for (auto v : A.col_idx) out.col_idx_i.push_back(static_cast<int>(v));

    return out;
  }

}  // namespace eigen_bridge
