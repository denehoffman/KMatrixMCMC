#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H
#pragma once
#define ARMA_NO_DEBUG

#include "Amplitude.hpp"
#include "DataReader.hpp"
#include <string>
#include <armadillo>

using namespace std;

class Likelihood {
public:
  // Constructor
  Likelihood(const string& data_path,
             const string& acc_path,
             const string& gen_path,
             const string& data_tree = "kin",
             const string& acc_tree = "kin",
             const string& gen_tree = "kin");

  // Setup function
  void setup();

  // Calculate log likelihood
  float getExtendedLogLikelihood(const vector<float>& params);

private:
  Amplitude amplitude;
  DataReader data;
  DataReader acc;
  DataReader gen;
  int nGenerated;
  void printLoadingBar (const int& progress, const int& total, const int& barWidth = 50) const;
  vector<arma::cx_fvec> ikc_inv_vec_f0;
  vector<arma::cx_fvec> ikc_inv_vec_f2;
  vector<arma::cx_fvec> ikc_inv_vec_a0;
  vector<arma::cx_fvec> ikc_inv_vec_a2;
  vector<arma::fmat> bw_f0;
  vector<arma::fmat> bw_f2;
  vector<arma::fmat> bw_a0;
  vector<arma::fmat> bw_a2;

  vector<arma::cx_fvec> ikc_inv_vec_f0_mc;
  vector<arma::cx_fvec> ikc_inv_vec_f2_mc;
  vector<arma::cx_fvec> ikc_inv_vec_a0_mc;
  vector<arma::cx_fvec> ikc_inv_vec_a2_mc;
  vector<arma::fmat> bw_f0_mc;
  vector<arma::fmat> bw_f2_mc;
  vector<arma::fmat> bw_a0_mc;
  vector<arma::fmat> bw_a2_mc;
};

#endif  // LIKELIHOOD_H
