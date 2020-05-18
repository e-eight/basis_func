/*******************************************************************************
 ho.h

 Computes all the radial harmonic oscillator wave functions up to a given radial
 quantum number.
 Ported from Kyle Wendt's Python code.

 Language: C++11
 Soham Pal
 Iowa State University
*******************************************************************************/

#ifndef HO_H_
#define HO_H_

#include <Eigen/Dense>
#include <vector>

#include "basis_func.h"

namespace basis_func {
namespace ho {
void WaveFunctionsUptoMaxN(Eigen::ArrayXXd& vals, const Eigen::ArrayXd& pts,
                           const std::size_t& max_n, const std::size_t& l,
                           const double& length, const Space& space);

void WaveFunctionsUptoMaxL(std::vector<Eigen::ArrayXXd>& wfs,
                           const Eigen::ArrayXd& pts, const std::size_t& max_n,
                           const std::size_t& max_l, const double& length,
                           const Space& space);
}  // namespace ho
}  // namespace basis_func

#endif
