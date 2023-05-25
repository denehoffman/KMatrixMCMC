#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

#include "Amplitude.h"
#include "DataReader.h"
#include <string>

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
  double getExtendedLogLikelihood();

private:
  Amplitude amplitude_;
  DataReader data_reader_;
  DataReader acc_reader_;
  DataReader gen_reader_;
};

#endif  // LIKELIHOOD_H
