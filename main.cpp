#include <iostream>
#include <armadillo>
#include <complex>
#include <chrono>
#include <sstream>
#include "KMatrix.h"
#include "Amplitude.h"
#include "Likelihood.h"
#include "DataReader.h"
#include "TH1F.h"
#include "TCanvas.h"

using namespace std;
using namespace arma;


int main(int argc, char* argv[]) {
  // std::vector<float> s_values;
  // for (size_t i = 1; i < argc; i++) {
  //   istringstream iss(argv[i]);
  //   float value;
  //   if (iss >> value) {
  //     s_values.push_back(value);
  //   } else {
  //     cout << "Invalid argument at index " << i << endl;
  //     return 1;
  //   }
  // }

  vector<float> params {
    1.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
  };
  cout << "Starting Calculation" << endl;
  auto startTime = chrono::high_resolution_clock::now();
  Likelihood lh("sandbox/data_short.root", "sandbox/accmc_short.root", "sandbox/accmc_short.root");
  auto endTime = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  cout << "Files loaded in " << duration << "ms" << endl;
  startTime = chrono::high_resolution_clock::now();
  lh.setup();
  endTime = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  cout << endl << "Precalculation took " << duration << "ms" << endl;
  startTime = chrono::high_resolution_clock::now();
  cout << lh.getExtendedLogLikelihood(params) << endl;
  endTime = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  cout << endl << "Amplitude calculation took " << duration << "ms" << endl;
  
  return 0;
}
