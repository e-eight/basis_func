/*******************************************************************************
 ho.h

 Computes all the radial harmonic oscillator wave functions up to a given n.
 Ported from Kyle Wendt's Python code.

 Language: C++11
 Soham Pal
 Iowa State University
*******************************************************************************/

#ifndef HO_H_
#define HO_H_

#include <string>
#include <Eigen/Dense>

namespace basis_func {
  namespace ho {

    void WF(Eigen::ArrayXXd& vals,
            const Eigen::ArrayXd& pts,
            const std::size_t& n,
            const std::size_t& l,
            const double& length,
            const std::string space);
  }
}

#endif
