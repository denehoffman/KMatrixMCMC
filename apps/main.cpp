#define ARMA_USE_HDF5
#include <iostream>
#include <armadillo>
#include <complex>
#include <chrono>
#include <sstream>
#include <tyche>
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

  cout << "Starting Calculation" << endl;
  Likelihood lh(argv[1], argv[2], argv[3]);
  lh.setup();
  std::function<float(const Col<float>&)> lambda_func = [&](const Col<float>& x) {
    return lh.getExtendedLogLikelihood(x);
  };
  tyche::Ensemble<float> ensemble(
    70,
      {
        // {"f0(980) Magnitude", 0.0f, 10000.0f},
        {"f0(1370) Magnitude", 0.0f, 1000.0f},
        {"f0(1370) Phase", 0.0f, arma::Datum<float>::tau},
        {"f0(1500) Magnitude", 0.0f, 1000.0f},
        {"f0(1500) Phase", 0.0f, arma::Datum<float>::tau},
        {"f0(1710) Magnitude", 0.0f, 1000.0f},
        {"f0(1710) Phase", 0.0f, arma::Datum<float>::tau},
        {"f2(1270) Magnitude", 0.0f, 1000.0f},
        {"f2(1270) Phase", 0.0f, arma::Datum<float>::tau},
        {"f2(1525) Magnitude", 0.0f, 1000.0f},
        {"f2(1525) Phase", 0.0f, arma::Datum<float>::tau},
        {"f2(1810) Magnitude", 0.0f, 1000.0f},
        {"f2(1810) Phase", 0.0f, arma::Datum<float>::tau},
        {"f2(1950) Magnitude", 0.0f, 1000.0f},
        {"f2(1950) Phase", 0.0f, arma::Datum<float>::tau},
        {"a0(980) Magnitude", 0.0f, 1000.0f},
        {"a0(980) Phase", 0.0f, arma::Datum<float>::tau},
        {"a0(1450) Magnitude", 0.0f, 1000.0f},
        {"a0(1450) Phase", 0.0f, arma::Datum<float>::tau},
        {"a2(1320) Magnitude", 0.0f, 1000.0f},
        {"a2(1320) Phase", 0.0f, arma::Datum<float>::tau},
        {"a2(1700) Magnitude", 0.0f, 1000.0f},
        {"a2(1700) Phase", 0.0f, arma::Datum<float>::tau}
      },
      lambda_func 
    );
  cout << "Setup done, initializing walkers" << endl;
  ensemble.init();
  cout << "Beginning MCMC" << endl;
  for (uint j = 0; j < 50; j++) {
    ensemble.sample({{new tyche::StretchMove<float>(), 0.5f},
                     {new tyche::DifferentialEvolutionMove<float>(23), 0.30f},
                     {new tyche::DifferentialEvolutionMove<float>(23, 1.0e-5, 1.0), 0.05f},
                     {new tyche::DifferentialEvolutionSnookerMove<float>(), 0.15f}});
    cout << j << endl;
  }
  cout << "Saving!" << endl;
  ensemble.save("MCMC.h5");

  // for (uint j = 0; j < 100; j++) {
  //   ensemble.sample({{new tyche::StretchMove<float>(), 0.5f},
  //                    {new tyche::DifferentialEvolutionMove<float>(23), 0.3f},
  //                    {new tyche::DifferentialEvolutionMove<float>(23, 1.0e-5, 1.0), 0.1f},
  //                    {new tyche::DifferentialEvolutionSnookerMove<float>(), 0.1f}});
  // }
  // cout << "Saving!" << endl;
  // ensemble.save("MCMC.h5");

  // for (uint j = 0; j < 200; j++) {
  //   ensemble.sample({{new tyche::StretchMove<float>(), 0.5f},
  //                    {new tyche::DifferentialEvolutionMove<float>(23), 0.3f},
  //                    {new tyche::DifferentialEvolutionSnookerMove<float>(), 0.2f}});
  // }
  // cout << "Saving!" << endl;
  // ensemble.save("MCMC.h5");
  return 0;
}
