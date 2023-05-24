#ifndef AMPLITUDE_H
#define AMPLITUDE_H

#include <armadillo>
#include "KMatrix.h"

using namespace arma;

class Amplitude {
  public:
    Amplitude();
    double intensity(const double& s, const cx_vec& betas);

  private:
    KMatrix kmat_f0;
    mat f0_mchannels;
    mat f0_malphas;
    mat f0_galphas;
    mat f0_cbkg;

    KMatrix kmat_f2;
    mat f2_mchannels;
    mat f2_malphas;
    mat f2_galphas;
    mat f2_cbkg;

    KMatrix kmat_a0;
    mat a0_mchannels;
    mat a0_malphas;
    mat a0_galphas;
    mat a0_cbkg;

    KMatrix kmat_a2;
    mat a2_mchannels;
    mat a2_malphas;
    mat a2_galphas;
    mat a2_cbkg;
};

#endif
