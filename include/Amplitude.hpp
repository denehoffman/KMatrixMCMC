#ifndef AMPLITUDE_H
#define AMPLITUDE_H
#pragma once
#define ARMA_NO_DEBUG
#include <armadillo>
#include "KMatrix.hpp"

using namespace arma;

class Amplitude {
  public:
    Amplitude();
    float intensity(const cx_fvec& betas, const float& s, const float& theta, const float& phi,
        const cx_fvec& ikc_inv_f0,
        const cx_fvec& ikc_inv_f2,
        const cx_fvec& ikc_inv_a0,
        const cx_fvec& ikc_inv_a2);
    float intensity(const cx_fvec& betas, const float& s, const float& theta, const float& phi,
        const fmat& bw_f0,
        const fmat& bw_f2,
        const fmat& bw_a0,
        const fmat& bw_a2,
        const cx_fvec& ikc_inv_f0,
        const cx_fvec& ikc_inv_f2,
        const cx_fvec& ikc_inv_a0,
        const cx_fvec& ikc_inv_a2);
    complex<float> S0_wave();
    complex<float> D2_wave(const float& theta, const float& phi);
    cx_fvec ikc_inv_vec_f0(const float& s);
    cx_fvec ikc_inv_vec_f2(const float& s);
    cx_fvec ikc_inv_vec_a0(const float& s);
    cx_fvec ikc_inv_vec_a2(const float& s);
    fmat bw_f2(const float& s);
    fmat bw_a2(const float& s);

  private:
    KMatrix kmat_f0 = KMatrix(5, 5, 0);
    fmat f0_mchannels;
    fmat f0_malphas;
    fmat f0_galphas;
    fmat f0_cbkg;

    KMatrix kmat_f2 = KMatrix(4, 4, 2);
    fmat f2_mchannels;
    fmat f2_malphas;
    fmat f2_galphas;
    fmat f2_cbkg;

    KMatrix kmat_a0 = KMatrix(2, 2, 0);
    fmat a0_mchannels;
    fmat a0_malphas;
    fmat a0_galphas;
    fmat a0_cbkg;

    KMatrix kmat_a2 = KMatrix(3, 2, 2);
    fmat a2_mchannels;
    fmat a2_malphas;
    fmat a2_galphas;
    fmat a2_cbkg;
};

#endif  // AMPLITUDE_H
