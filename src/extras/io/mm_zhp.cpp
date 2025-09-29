/**
 * \file mm_zhp.cpp
 * \brief Implementation of load_mm_zhp (Matrix Market -> BLAS ZHP).
 * \details Indexing (ZHP):
 *           U: k = j*(j+1)/2 + i  (0 ≤ i ≤ j < n)
 *           L: k = i*(i+1)/2 + j  (0 ≤ j ≤ i < n)
 */

#include "gsi_sminres/extras/io/mm_zhp.hpp"

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gsi_sminres {
  namespace io {

    std::vector<std::complex<double>> load_mm_zhp(const std::string& filename,
                                                  std::size_t&       n,
                                                  char&              uplo) {
      // Referenced: https://math.nist.gov/MatrixMarket/mmio-c.html
      // Open file
      std::ifstream ifs(filename);
      if (!ifs)
        throw std::runtime_error("load_mm_zhp: [ERROR] cannot open: " + filename);
      // Analyze header
      std::string line;
      if (!std::getline(ifs, line))
        throw std::runtime_error("load_mm_zhp: [ERROR] empty file");
      std::string h = line;
      std::transform(h.begin(), h.end(), h.begin(),
                     [](unsigned char c){
                       return static_cast<char>(std::tolower(c));
                     }); // 大文字から小文字に
      if (h.find("%%matrixmarket matrix coordinate") == std::string::npos)
        throw std::runtime_error("load_mm_zhp: [ERROR] not a coordinate Matrix Market file");
      const bool is_real      = (h.find("real")      != std::string::npos);
      const bool is_complex   = (h.find("complex")   != std::string::npos);
      const bool is_symmetric = (h.find("symmetric") != std::string::npos);
      const bool is_hermitian = (h.find("hermitian") != std::string::npos);
      if (!((is_real && is_symmetric) || (is_complex && is_hermitian)))
        throw std::runtime_error("load_mm_zhp: [ERROR] supported kinds: real symmetric OR complex hermitian");
      // Read size line (skip comments/blank)
      while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
      }
      if (ifs.fail())
        throw std::runtime_error("load_mm_zhp: [ERROR] missing size line");
      std::istringstream iss(line);
      std::size_t nrows = 0, ncols = 0, nvals = 0;
      if (!(iss >> nrows >> ncols >> nvals))
        throw std::runtime_error("load_mm_zhp: [ERROR] bad size line");
      if (nrows != ncols)
        throw std::runtime_error("load_mm_zhp: [ERROR] matrix must be square");
      n = nrows;
      // Allocate packed array (n(n+1)/2)
      if (n > static_cast<std::size_t>(-1) / (n + 1) / 2)
        throw std::runtime_error("load_mm_zhp: [ERROR] size too large for packed allocation");
      std::vector<std::complex<double>> packed(n * (n + 1) / 2, {0.0, 0.0});
      // Set uplo
      uplo = 'U';
      // Read matrix elements
      std::size_t row, col;
      double real, imag;
      for (std::size_t t = 0; t < nvals; ++t) {
        if (is_real && is_symmetric) {
          if (!(ifs >> row >> col >> real)) {
            throw std::runtime_error("load_mm_zhp: [ERROR] Invalid matrix elements in " + filename);
          }
          row -= 1; col -= 1;
          if (row <= col) {
            packed[row + col*(col+1)/2] += std::complex<double>(real, 0.0);
          } else {
            packed[col + row*(row+1)/2] += std::complex<double>(real, 0.0);
          }
        } else if (is_complex && is_hermitian) {
          if (!(ifs >> row >> col >> real >> imag)) {
            throw std::runtime_error("load_mm_zhp: [ERROR] Invalid matrix elements in " + filename);
          }
          row -= 1; col -= 1;
          if (row <= col) {
            packed[row + col*(col+1)/2] += std::complex<double>(real, imag);
          } else {
            packed[col + row*(row+1)/2] += std::complex<double>(real, -imag);
          }
        } else {
          throw std::runtime_error("load_mm_zhp: [ERROR] Invalid matrix format in " + filename);
        }
      }
      // Return packed matrix
      return packed;
    }

  } // namespace io
} // namespace gsi_sminres
