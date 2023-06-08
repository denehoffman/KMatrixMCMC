#ifndef KMATRIX_H
#define KMATRIX_H
#pragma once
#define ARMA_NO_DEBUG

#include <unordered_map>
#include <vector>
#include <iostream>
#include <armadillo>
#include <complex>

using namespace std;

/**
 * @brief The K-Matrix class
 */
class KMatrix {
public:
    int numAlphas;
    int numChannels;
    int J;
    arma::cx_fmat mAlphas;
    arma::cx_fmat mChannels;
    arma::cx_fvec m1s;
    arma::cx_fvec m2s;
    arma::cx_fmat gAlphas;
    arma::cx_fmat cBkg;
    arma::cx_fmat bwAlphaMat;
    arma::cx_fcube bwAlphaCube;

    // Constructor
    KMatrix(int numChannels, int numAlphas, int J);

    // Initialize the matrices and vectors
    void initialize(
        const arma::fmat& m_alphas,
        const arma::fmat& m_channels,
        const arma::fmat& g_alphas,
        const arma::fmat& c_bkg);

    // Print the matrices and vectors
    void print() const;

    arma::cx_fvec chi_p(const float& s) const;
    arma::cx_fvec chi_m(const float& s) const;
    arma::cx_fvec rho(const float& s) const;
    arma::cx_fvec q(const float& s) const;
    arma::cx_fvec blatt_weisskopf(const float& s) const;
    arma::cx_fmat B(const float& s) const;
    arma::cx_fcube B2(const float& s) const;
    arma::cx_fmat K(const float& s) const;
    arma::cx_fmat K(const float& s, const float& s_0, const float& s_norm) const;
    arma::cx_fmat C(const float& s) const;
    arma::cx_fmat IKC_inv(const float& s);
    arma::cx_fmat IKC_inv(const float& s, const float& s_0, const float& s_norm);
    arma::cx_fvec P(const float& s, const arma::cx_fvec& betas) const;
    arma::cx_fvec P(const float& s, const arma::cx_fvec& betas, const arma::cx_fmat& B) const;
    complex<float> F(const float& s, const arma::cx_fvec& betas, const arma::cx_fvec& ikc_inv_vec);
    complex<float> F(const float& s, const arma::cx_fvec& betas, const arma::cx_fmat& B, const arma::cx_fvec& ikc_inv_vec);
  
  private:
    function<arma::cx_fvec(const float&)> blattWeisskopfPtr;
    arma::cx_fvec blatt_weisskopf0(const float& s);
    arma::cx_fvec blatt_weisskopf2(const float& s);
};

#endif  // KMATRIX_H
