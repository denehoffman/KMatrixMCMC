#include "Likelihood.h"
#include "Amplitude.h"
#include "DataReader.h"

Likelihood::Likelihood(const string& data_path,
                       const string& acc_path,
                       const string& gen_path,
                       const string& data_tree,
                       const string& acc_tree,
                       const string& gen_tree)
  : amplitude_(),
  data_reader_(data_path, data_tree),
  acc_reader_(acc_path, acc_tree),
  gen_reader_(gen_path, gen_tree) {}

void Likelihood::setup() {
  // Perform any necessary setup steps here
}

double Likelihood::getExtendedLogLikelihood() {
  // Perform likelihood calculation and return the result
  double log_likelihood = 0.0;
  // ... calculation logic goes here ...
  return log_likelihood;
}
