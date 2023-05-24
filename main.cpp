#include <iostream>
#include <armadillo>
#include <complex>
#include <sstream>
#include <TGraph.h>
#include <TCanvas.h>
#include "KMatrix.h"

using namespace std;
using namespace arma;


int main(int argc, char* argv[]) {
  std::vector<double> s_values;
  for (size_t i = 1; i < argc; i++) {
    istringstream iss(argv[i]);
    double value;
    if (iss >> value) {
      s_values.push_back(value);
    } else {
      cout << "Invalid argument at index " << i << endl;
      return 1;
    }
  }
  KMatrix kmat_a2(3, 2);
  mat a2_mChannels = {
    {0.13498, 0.54786},  // pi eta
    {0.49368, 0.49761},  // K K
    {0.13498, 0.95778}}; // pi eta'
  //               a2  1320      1700
  mat a2_mAlphas = {1.30080,  1.75351};
  // pi eta       K K      pi eta'
  mat a2_gAlphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  mat a2_cBkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}};
  kmat_a2.initialize(a2_mAlphas, a2_mChannels, a2_gAlphas.t(), a2_cBkg, 2);
  kmat_a2.print();

  // Test chi_p function
  for (const double& s : s_values) {
    arma::cx_mat ikc_inv = kmat_a2.IKC_inv(s);
    arma::cx_vec betas = {1.0, 2.0};
    cout << pow(abs(kmat_a2.F(s, betas, ikc_inv, 2)), 2) << endl;
  }

  return 0;
}
