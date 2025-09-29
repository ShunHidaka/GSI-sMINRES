/**
 * \file gsi_sminres.hpp
 * \brief Umbrella header including all public GSI-MINRES solver APIs.
 * \author Shuntaro Hidaka
 *
 * \details Includes:
 *           - generalized_shifted_minres.hpp
 *           - shift_invert_shifted_minres.hpp
 *           - standard_shifted_minres.hpp
 */

#ifndef GSI_SMINRES_GSI_MINRES_HPP
#define GSI_SMINRES_GSI_MINRES_HPP

#include <gsi_minres/version.hpp>

#include <gsi_minres/algorithms/generalized_shifted_minres.hpp>
#include <gsi_minres/algorithms/shift_invert_shifted_minres.hpp>
#include <gsi_minres/algorithms/standard_shifted_minres.hpp>


/**
 * \namespace gsi_sminres
 * \brief Top-Level namespace of the GSI-MINRES library
 *
 * \details Major sub-namespaces:
 *  - \c generalized           : generalized shifted minres solver interface
 *  - \c shift_invert          : shift-invert precondition shifted minres solver interface
 *  - \c standard              : standard shifted minres solver interface
 *  - \c linalg::{blas,lapack} : linear algebra wrappers
 *  - \c io::{mm,csr}          : I/O utilities for easy-to-use GSI-MINRES
 *  - \c extras                : CG method and MINRES method
 */
namespace gsi_sminres {

  /**
   * \namespace gsi_sminres::linalg
   * \brief [Utility] BLAS / LAPACK wrappers
   */
  namespace linalg {
    
  }

  /**
   * \namespace gsi_sminres::io
   * \brief [Utility] I/O utilities for sample codes (Matrix Market converters).
   */
  namespace io {
    
  }

  /**
   * \namespace gsi_sminres::extras
   * \brief [Utility] linear solvers for sample programs.
   */
  namespace extras {
    
  }
}


#endif // GSI_SMINRES_GSI_MINRES_HPP
