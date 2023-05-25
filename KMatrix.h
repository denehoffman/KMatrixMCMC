#ifndef KMATRIX_H
#define KMATRIX_H

#include <unordered_map>
#include <vector>
#include <iostream>
#include <armadillo>
#include <complex>

using namespace std;

class KMatrix {
public:
    int numAlphas;
    int numChannels;
    int J;
    arma::cx_mat mAlphas;
    arma::cx_mat mChannels;
    arma::cx_mat gAlphas;
    arma::cx_mat cBkg;

    // Constructor
    KMatrix(int numChannels, int numAlphas);

    // Initialize the matrices and vectors
    void initialize(
        const arma::mat& m_alphas,
        const arma::mat& m_channels,
        const arma::mat& g_alphas,
        const arma::mat& c_bkg,
        const int& j);

    // Print the matrices and vectors
    void print() const;
    void clearCache();

    arma::cx_vec chi_p(const double& s) const;
    arma::cx_vec chi_m(const double& s) const;
    arma::cx_vec rho(const double& s) const;
    arma::cx_vec q(const double& s) const;
    arma::cx_vec blatt_weisskopf(const double& s) const;
    arma::cx_mat B(const double& s) const;
    arma::cx_cube B2(const double& s) const;
    arma::cx_mat K(const double& s) const;
    arma::cx_mat K(const double& s, const double& s_0, const double& s_norm) const;
    arma::cx_mat C(const double& s) const;
    arma::cx_mat IKC_inv(const double& s);
    arma::cx_mat IKC_inv(const double& s, const double& s_0, const double& s_norm);
    arma::cx_vec P(const double& s, const arma::cx_vec& betas) const;
    complex<double> F(const double& s, const arma::cx_vec& betas, const int& channel);
    complex<double> F(const double& s, const arma::cx_vec& betas, const double& s_0, const double& s_norm, const int& channel);

  private:
    unordered_map<double, arma::cx_mat> cache;
};

#endif  // KMATRIX_H
