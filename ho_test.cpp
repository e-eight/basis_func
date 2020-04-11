#include <fstream>
#include <iostream>
#include "ho.h"

int main() {
  int npts = 3000;
  int n = 1000;
  int l = 0;
  double b = 1.0;

  Eigen::ArrayXXd val(n+1, npts);
  Eigen::ArrayXd pts = Eigen::ArrayXd::LinSpaced(npts, 0, 10);
  try {
    basis_func::ho::WF(val, pts, n, l, b, "coordinate");
  }
  catch (std::exception& e) {
    std::cout << e.what() << "\n";
  }

  Eigen::ArrayXXd output(npts, 2);
  output.col(0) = pts;
  output.col(1) = val.row(n);
  Eigen::IOFormat fmt(7, Eigen::DontAlignCols, " ", "\n");

  std::ofstream file;
  file.open("ho_wf.txt");
  file << output.format(fmt) << "\n";
  file.close();
}
