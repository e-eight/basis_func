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
            const std::string space) {
      if (space != "coordinate" && space != "momentum") {
        throw std::runtime_error("space must be coordinate or momentum.");
      }

      int npts = pts.size();
      int ni = static_cast<int>(n);
      int li = static_cast<int>(l);

      if (vals.rows() != ni + 1 || vals.cols() != npts) {
        throw std::runtime_error("vals must have the dimensions (n + 1, pts.size())");
      }

      Eigen::ArrayXd pts2(npts);
      pts2 = Eigen::square(pts);
      if (space == "coordinate") {
        pts2 /= length * length;
      } else if (space == "momentum") {
        pts2 *= length * length;
      }

      double jac = 0;
      if (space == "coordinate") {
        jac = 1.0 / std::pow(length, 1.5);
      } else if (space == "momentum") {
        jac = std::pow(length, 1.5);
      }

      Eigen::ArrayXd psi_i(npts), psi_im1(npts), psi_im2(npts);
      // n = 0
      psi_im2 = (jac * Eigen::pow(pts, li)
                 * std::sqrt(2.0 / std::tgamma(li + 1.5))
                 * Eigen::exp(-0.5 * pts2));
      // n = 1
      psi_im1 = (-jac * Eigen::pow(pts, li)
                 * std::sqrt(2.0 / std::tgamma(li + 2.5))
                 * (1.5 + li - pts2)
                 * Eigen::exp(-0.5 * pts2));

      vals.row(0) = psi_im2;
      vals.row(1) = psi_im1;

      for (int i = 2; i <= ni; ++i) {
        psi_i = (-(2 * i + li - 0.5 - pts2) * std::sqrt(1 / (i * (i + li + 0.5))) * psi_im1
                 - std::sqrt((i + li - 0.5) * (i - 1) / (i * (i + li + 0.5))) * psi_im2);
        psi_im2 = psi_im1;
        psi_im1 = psi_i;
        vals.row(i) = psi_i;
      }
    }
  }
}
