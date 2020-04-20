#include <fstream>
#include <iostream>
#include <iomanip>
#include "ho.h"

int main() {
  int npts = 3001;
  int n = 10;
  int l = 20;
  double b = 2.03644;

  // Array for wavefunctions.
  Eigen::ArrayXXd psi(n + 1, npts);

  // Map [0 to ∞) to [0 to 1)
  Eigen::ArrayXd x = Eigen::ArrayXd::LinSpaced(npts, 0, 1.0);
  Eigen::ArrayXd r = x / (1.0 - x);
  Eigen::ArrayXd dr = 1 / ((1.0 - x) * (1.0 - x));

  try {
    basis_func::ho::WF(psi, r, n, l, b, basis_func::Space::coordinate);
  }
  catch (std::exception& e) {
    std::cout << e.what() << "\n";
  }

  Eigen::ArrayXXd output(npts - 1, 2);
  output.col(0) = r.head(npts - 1);
  output.col(1) = psi.row(n).head(npts - 1);
  Eigen::IOFormat fmt(7, Eigen::DontAlignCols, " ", "\n");

  std::ofstream file;
  file.open("ho_wf.txt");
  file << output.format(fmt) << "\n";
  file.close();

  // Wavefunction norm.

  Eigen::ArrayXd psin2 = psi.row(n).square();
  psin2 *= r * r * dr;
  psin2.head(1) *= 0.5;
  psin2.tail(1) = 0;
  double norm_nn = psin2.sum() / npts;

  Eigen::ArrayXd psinn1 = psi.row(n) * psi.row(n-1);
  psinn1 *= r * r * dr;
  psinn1.head(1) *= 0.5;
  psinn1.tail(1) = 0;
  double norm_nn1 = psinn1.sum() / npts;

  std::cout << "∫R_{nl} R_{nl} r^2 dr = " << std::setprecision(15) << std::fixed << norm_nn << "\n";
  std::cout << "∫R_{nl} R_{n-1l} r^2 dr = " << norm_nn1 << "\n";

  // Check if all the wave functions are normalized by matrix multiplication.
  // Diagonal elements should be ~1. Off-diagonal elements should be ~0.
  // Trapezoid weights.
  Eigen::ArrayXd wt = Eigen::ArrayXd::Ones(npts);
  wt.head(1) *= 0.5;
  wt.tail(2) *= 0.5;
  wt /= npts;

  psi.col(npts - 1) = 0;
  Eigen::ArrayXXd right_psi(n + 1, npts);
  right_psi = psi.transpose().colwise() * (wt * r.square() * dr);
  right_psi.row(npts - 1) = 0;

  Eigen::ArrayXXd norms(n + 1, npts);
  norms = (psi.matrix() * right_psi.matrix()).array();
  std::cout << norms << "\n";
}
