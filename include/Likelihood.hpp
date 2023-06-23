#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H
#pragma once
// #define ARMA_NO_DEBUG

#include "Amplitude.hpp"
#include "DataReader.hpp"
#include <string>
#include <armadillo>
#include <vector>
#include <deque>

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
  float getExtendedLogLikelihood(const arma::Col<float>& params);

private:
  Amplitude amplitude;
  DataReader data;
  DataReader acc;
  DataReader gen;
  int nGenerated;
  void printLoadingBar (const int& progress, const int& total, const int& barWidth = 50) const;
  deque<arma::cx_fvec> ikc_inv_vec_f0;
  deque<arma::cx_fvec> ikc_inv_vec_f2;
  deque<arma::cx_fvec> ikc_inv_vec_a0;
  deque<arma::cx_fvec> ikc_inv_vec_a2;
  deque<arma::fmat> bw_f0;
  deque<arma::fmat> bw_f2;
  deque<arma::fmat> bw_a0;
  deque<arma::fmat> bw_a2;

  deque<arma::cx_fvec> ikc_inv_vec_f0_mc;
  deque<arma::cx_fvec> ikc_inv_vec_f2_mc;
  deque<arma::cx_fvec> ikc_inv_vec_a0_mc;
  deque<arma::cx_fvec> ikc_inv_vec_a2_mc;
  deque<arma::fmat> bw_f0_mc;
  deque<arma::fmat> bw_f2_mc;
  deque<arma::fmat> bw_a0_mc;
  deque<arma::fmat> bw_a2_mc;
};

#endif  // LIKELIHOOD_H
