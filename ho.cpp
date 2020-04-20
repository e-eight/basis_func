#include <cmath>
#include <stdexcept>
#include "ho.h"

namespace basis_func {
  namespace ho {

    void WF(Eigen::ArrayXXd& vals,
            const Eigen::ArrayXd& pts,
            const std::size_t& n,
            const std::size_t& l,
            const double& length,
            const Space& space) {
      if (space != Space::coordinate && space != Space::momentum) {
        throw std::runtime_error("space must be coordinate or momentum.");
      }

      int npts = pts.size();
      int ni = static_cast<int>(n);
      int li = static_cast<int>(l);

      if (vals.rows() != ni + 1 || vals.cols() != npts) {
        throw std::runtime_error("vals must have the dimensions (n + 1, pts.size())");
      }

      Eigen::ArrayXd x(npts), x2(npts);
      x = pts;
      if (space == Space::coordinate) {
        x /= length;
      } else if (space == Space::momentum) {
        x *= length;
      }
      x2 = x.square();

      double jac = 0;
      if (space == Space::coordinate) {
        jac = std::pow(length, -1.5);
      } else if (space == Space::momentum) {
        jac = std::pow(length, 1.5);
      }

      Eigen::ArrayXd psi_i(npts), psi_im1(npts), psi_im2(npts);
      // n = 0
      psi_im2 = (jac * Eigen::pow(x, li)
                 * std::sqrt(2.0 / std::tgamma(li + 1.5))
                 * Eigen::exp(-0.5 * x2));
      // n = 1
      psi_im1 = (-jac * Eigen::pow(x, li)
                 * std::sqrt(2.0 / std::tgamma(li + 2.5))
                 * (1.5 + li - x2)
                 * Eigen::exp(-0.5 * x2));

      vals.row(0) = psi_im2;
      vals.row(1) = psi_im1;

      for (int i = 2; i <= ni; ++i) {
        psi_i = ((-std::sqrt(2. * i / (1. + 2. * (li + i))) * (2. + (li - 0.5 - x2) / i) * psi_im1)
                 - (std::sqrt(4. * i * (i - 1) / (4. * (i + li) * (i + li) - 1.)) * (1. + (li - 0.5) / i) * psi_im2));
        psi_im2 = psi_im1;
        psi_im1 = psi_i;
        vals.row(i) = psi_i;
      }
    }
  }
}
