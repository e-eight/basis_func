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

#include <Eigen/Dense>
#include "space.h"

namespace basis_func {
  namespace ho {

    void WF(Eigen::ArrayXXd& vals,
            const Eigen::ArrayXd& pts,
            const std::size_t& n,
            const std::size_t& l,
            const double& length,
            const Space& space);
  }
}

#endif
