#ifndef AMPLITUDE_H
#define AMPLITUDE_H

#include <armadillo>
#include "KMatrix.h"

using namespace arma;

class Amplitude {
  public:
    Amplitude();
    double intensity(const cx_vec& betas, const double& s, const double& theta, const double& phi,
        const cx_mat& ikc_inv_f0,
        const cx_mat& ikc_inv_f2,
        const cx_mat& ikc_inv_a0,
        const cx_mat& ikc_inv_a2);
    complex<double> S0_wave(const double& theta, const double& phi);
    complex<double> D2_wave(const double& theta, const double& phi);
    cx_vec ikc_inv_vec_f0(const double& s);
    cx_vec ikc_inv_vec_f2(const double& s);
    cx_vec ikc_inv_vec_a0(const double& s);
    cx_vec ikc_inv_vec_a2(const double& s);

  private:
    KMatrix kmat_f0 = KMatrix(5, 5, 0);
    mat f0_mchannels;
    mat f0_malphas;
    mat f0_galphas;
    mat f0_cbkg;

    KMatrix kmat_f2 = KMatrix(4, 4, 2);
    mat f2_mchannels;
    mat f2_malphas;
    mat f2_galphas;
    mat f2_cbkg;

    KMatrix kmat_a0 = KMatrix(2, 2, 0);
    mat a0_mchannels;
    mat a0_malphas;
    mat a0_galphas;
    mat a0_cbkg;

    KMatrix kmat_a2 = KMatrix(3, 2, 2);
    mat a2_mchannels;
    mat a2_malphas;
    mat a2_galphas;
    mat a2_cbkg;
};

#endif
