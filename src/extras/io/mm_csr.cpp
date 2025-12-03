/**
 * \file mm_csr.cpp
 * \brief Implementation of load_mm_csr (Matrix Market →CSRMatrix).
 */

#include "gsi_sminres/extras/io/mm_csr.hpp"

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gsi_sminres {
  namespace io {

    namespace {
      [[noreturn]] void die(const std::string& msg) {
        std::cerr << msg << '\n';
        std::exit(EXIT_FAILURE);
      }
      struct Entry {
        std::size_t i, j;       // (row, col) = (i, j) : 0-based
        std::complex<double> v; // value
      };
      struct Entry_r {
        std::size_t i, j;
        double v;
      };
      inline bool by_row_col(const Entry& a, const Entry& b) {
        return (a.i < b.i) || (a.i == b.i && a.j < b.j);
      }
      inline bool by_row_col_r(const Entry_r& a, const Entry_r& b) {
        return (a.i < b.i) || (a.i == b.i && a.j < b.j);
      }
    } // anonymous namespace

    gsi_sminres::sparse::CSRMatrix load_mm_csr(const std::string& filename,
                                               std::size_t&       n) {
      std::ifstream fin(filename);
      if (!fin) die("load_mm_csr: [ERROR] cannot open: " + filename);

      // ---- Header (case-insensitive) ----
      std::string line;
      if (!std::getline(fin, line)) die("load_mm_csr: [ERROR] empty file");
      std::string h = line;
      std::transform(h.begin(), h.end(), h.begin(),
                     [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
      if (h.find("%%matrixmarket matrix coordinate") == std::string::npos)
        die("load_mm_csr: [ERROR] not a coordinate Matrix Market file");

      const bool is_real      = (h.find(" real")      != std::string::npos);
      const bool is_complex   = (h.find(" complex")   != std::string::npos);
      const bool is_symmetric = (h.find(" symmetric") != std::string::npos);
      const bool is_hermitian = (h.find(" hermitian") != std::string::npos);

      if (!((is_real && is_symmetric) || (is_complex && is_hermitian)))
        die("load_mm_csr: [ERROR] supported kinds: real symmetric OR complex hermitian");

      // ---- Size line (skip comments/blank) ----
      while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
      }
      if (fin.fail()) die("load_mm_csr: [ERROR] missing size line");

      std::istringstream iss(line);
      std::size_t nrows = 0, ncols = 0, nnz_decl = 0;
      if (!(iss >> nrows >> ncols >> nnz_decl)) die("load_mm_csr: [ERROR] bad size line");
      if (nrows != ncols) die("load_mm_csr: [ERROR] matrix must be square");
      n = nrows;

      // ---- Read entries, expand symmetry/Hermitian to full (mirror) ----
      std::vector<Entry> coo;
      coo.reserve(nnz_decl * 2); // worst-case: all off-diagonals mirrored

      for (std::size_t k = 0; k < nnz_decl; ++k) {
        std::size_t ir=0, jc=0;
        double re=0.0, im=0.0;
        if (is_complex) {
          if (!(fin >> ir >> jc >> re >> im))
            die("load_mm_csr: [ERROR] invalid complex entry at t=" + std::to_string(k));
        } else {
          if (!(fin >> ir >> jc >> re))
            die("load_mm_csr: [ERROR] invalid real entry at t=" + std::to_string(k));
          im = 0.0;
        }
        if (ir==0 || jc==0) die("load_mm_csr: [ERROR] indices must be 1-based");
        std::size_t i = ir - 1, j = jc - 1;
        if (i >= n || j >= n) die("load_mm_csr: [ERROR] index out of range");

        const std::complex<double> v(re, im);
        coo.push_back({i, j, v});
        if (i != j) {
          if (is_symmetric) coo.push_back({j, i, v});
          else              coo.push_back({j, i, std::conj(v)}); // Hermitian
        }
      }

      // ---- Sort by (row, col) ----
      std::sort(coo.begin(), coo.end(), by_row_col);

      // ---- Merge duplicates and count per row ----
      gsi_sminres::sparse::CSRMatrix A;
      A.n = n;
      A.row_ptr.assign(n + 1, 0);

      std::vector<std::size_t> cols;
      std::vector<std::complex<double>> vals;
      cols.reserve(coo.size());
      vals.reserve(coo.size());

      std::size_t p = 0;
      while (p < coo.size()) {
        const auto row = coo[p].i;
        const auto col = coo[p].j;
        std::complex<double> sum = coo[p].v;
        ++p;
        while (p < coo.size() && coo[p].i == row && coo[p].j == col) {
          sum += coo[p].v;
          ++p;
        }
        ++A.row_ptr[row];
        cols.push_back(col);
        vals.push_back(sum);
      }

      // ---- Prefix sum (row_ptr) ----
      {
        std::size_t acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t cnt = A.row_ptr[i];
          A.row_ptr[i] = acc;
          acc += cnt;
          if (i + 1 == n) A.row_ptr[n] = acc;
        }
      }

      // ---- Scatter into CSR arrays (already row/col-sorted) ----
      const std::size_t nnz = cols.size();
      A.col_idx.resize(nnz);
      A.values.resize(nnz);

      {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t b = A.row_ptr[i];
          const std::size_t e = A.row_ptr[i + 1];
          for (std::size_t pos = b; pos < e; ++pos, ++idx) {
            A.col_idx[pos] = cols[idx];
            A.values[pos]  = vals[idx];
          }
        }
      }

      return A;
    }

    gsi_sminres::sparse::CSRMatrix_r load_mm_csr_r(const std::string& filename,
                                                   std::size_t&       n) {
      std::ifstream fin(filename);
      if (!fin) die("load_mm_csr: [ERROR] cannot open: " + filename);

      // ---- Header (case-insensitive) ----
      std::string line;
      if (!std::getline(fin, line)) die("load_mm_csr_r: [ERROR] empty file");
      std::string h = line;
      std::transform(h.begin(), h.end(), h.begin(),
                     [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
      if (h.find("%%matrixmarket matrix coordinate") == std::string::npos)
        die("load_mm_csr_r: [ERROR] not a coordinate Matrix Market file");

      const bool is_real      = (h.find(" real")      != std::string::npos);
      const bool is_symmetric = (h.find(" symmetric") != std::string::npos);

      if (!(is_real && is_symmetric))
        die("load_mm_csr_r: [ERROR] supported kinds: real symmetric");

      // ---- Size line (skip comments/blank) ----
      while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
      }
      if (fin.fail()) die("load_mm_csr_r: [ERROR] missing size line");

      std::istringstream iss(line);
      std::size_t nrows = 0, ncols = 0, nnz_decl = 0;
      if (!(iss >> nrows >> ncols >> nnz_decl)) die("load_mm_csr_r: [ERROR] bad size line");
      if (nrows != ncols) die("load_mm_csr_r: [ERROR] matrix must be square");
      n = nrows;

      // ---- Read entries, expand symmetry/Hermitian to full (mirror) ----
      std::vector<Entry_r> coo;
      coo.reserve(nnz_decl * 2); // worst-case: all off-diagonals mirrored

      for (std::size_t k = 0; k < nnz_decl; ++k) {
        std::size_t ir=0, jc=0;
        double re=0.0;
        if (!(fin >> ir >> jc >> re))
            die("load_mm_csr_r: [ERROR] invalid real entry at t=" + std::to_string(k));
        if (ir==0 || jc==0) die("load_mm_csr_r: [ERROR] indices must be 1-based");
        std::size_t i = ir - 1, j = jc - 1;
        if (i >= n || j >= n) die("load_mm_csr_r: [ERROR] index out of range");

        const double v = re;
        coo.push_back({i, j, v});
        if (i != j) {
          coo.push_back({j, i, v});
        }
      }

      // ---- Sort by (row, col) ----
      std::sort(coo.begin(), coo.end(), by_row_col_r);

      // ---- Merge duplicates and count per row ----
      gsi_sminres::sparse::CSRMatrix_r A;
      A.n = n;
      A.row_ptr.assign(n + 1, 0);

      std::vector<std::size_t> cols;
      std::vector<double> vals;
      cols.reserve(coo.size());
      vals.reserve(coo.size());

      std::size_t p = 0;
      while (p < coo.size()) {
        const auto row = coo[p].i;
        const auto col = coo[p].j;
        double sum     = coo[p].v;
        ++p;
        while (p < coo.size() && coo[p].i == row && coo[p].j == col) {
          sum += coo[p].v;
          ++p;
        }
        ++A.row_ptr[row];
        cols.push_back(col);
        vals.push_back(sum);
      }

      // ---- Prefix sum (row_ptr) ----
      {
        std::size_t acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t cnt = A.row_ptr[i];
          A.row_ptr[i] = acc;
          acc += cnt;
          if (i + 1 == n) A.row_ptr[n] = acc;
        }
      }

      // ---- Scatter into CSR arrays (already row/col-sorted) ----
      const std::size_t nnz = cols.size();
      A.col_idx.resize(nnz);
      A.values.resize(nnz);

      {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t b = A.row_ptr[i];
          const std::size_t e = A.row_ptr[i + 1];
          for (std::size_t pos = b; pos < e; ++pos, ++idx) {
            A.col_idx[pos] = cols[idx];
            A.values[pos]  = vals[idx];
          }
        }
      }

      return A;
    }

  } // namespace io
} // namespace gsi_sminres
