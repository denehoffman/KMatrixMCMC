#include <iostream>
#include <armadillo>
#include <complex>
#include <chrono>
#include <sstream>
#include "KMatrix.hpp"
#include "Amplitude.hpp"
#include "Likelihood.hpp"
#include "DataReader.hpp"
#include "TH1F.h"
#include "TCanvas.h"

using namespace std;
using namespace arma;


int main(int argc, char* argv[]) {
  if (argc < 4) {
    cout << "Insufficient command-line arguments provided." << endl;
    return 1;
  }

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
  Likelihood lh(argv[1], argv[2], argv[3]);
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
