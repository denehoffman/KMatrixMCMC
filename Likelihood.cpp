#include "Likelihood.h"
#include "Amplitude.h"
#include "DataReader.h"
#include <cmath>

Likelihood::Likelihood(const string& data_path,
                       const string& acc_path,
                       const string& gen_path,
                       const string& data_tree,
                       const string& acc_tree,
                       const string& gen_tree)
  : amplitude(),
  data(data_path, data_tree),
  acc(acc_path, acc_tree),
  gen(gen_path, gen_tree) {
    data.read();
    acc.read();
    nGenerated = gen.nEvents;
  }

void Likelihood::setup() {
  // Perform any necessary setup steps here
  cout << "Precalculating inverse of (I - KC)" << endl;
  cout << "Data" << endl;
  for (size_t i = 0; i < data.nEvents; i++) {
    printLoadingBar(i, data.nEvents);
    ikc_inv_vec_f0.push_back(amplitude.ikc_inv_vec_f0(pow(data.masses[i], 2)));
    ikc_inv_vec_f2.push_back(amplitude.ikc_inv_vec_f2(pow(data.masses[i], 2)));
    ikc_inv_vec_a0.push_back(amplitude.ikc_inv_vec_a0(pow(data.masses[i], 2)));
    ikc_inv_vec_a2.push_back(amplitude.ikc_inv_vec_a2(pow(data.masses[i], 2)));
    bw_f2.push_back(amplitude.bw_f2(pow(data.masses[i], 2)));
    bw_a2.push_back(amplitude.bw_a2(pow(data.masses[i], 2)));
  }
  cout << "Monte Carlo" << endl;
  for (size_t i = 0; i < acc.nEvents; i++) {
    printLoadingBar(i, acc.nEvents);
    ikc_inv_vec_f0_mc.push_back(amplitude.ikc_inv_vec_f0(pow(acc.masses[i], 2)));
    ikc_inv_vec_f2_mc.push_back(amplitude.ikc_inv_vec_f2(pow(acc.masses[i], 2)));
    ikc_inv_vec_a0_mc.push_back(amplitude.ikc_inv_vec_a0(pow(acc.masses[i], 2)));
    ikc_inv_vec_a2_mc.push_back(amplitude.ikc_inv_vec_a2(pow(acc.masses[i], 2)));
    bw_f2_mc.push_back(amplitude.bw_f2(pow(acc.masses[i], 2)));
    bw_a2_mc.push_back(amplitude.bw_a2(pow(acc.masses[i], 2)));
  }
}

float Likelihood::getExtendedLogLikelihood(const vector<float>& params) {
  // Perform likelihood calculation and return the result
  assert(params.size() == 23);
  cx_fvec betas = {
    polar<float>(0.0, 0.0),                // f0(500)
    polar<float>(params[0], 0.0),          // f0(980)
    polar<float>(params[1], params[2]),    // f0(1370)
    polar<float>(params[3], params[4]),    // f0(1500)
    polar<float>(params[5], params[6]),    // f0(1710)
    polar<float>(params[7], params[8]),    // f2(1270)
    polar<float>(params[9], params[10]),   // f2(1525)
    polar<float>(params[11], params[12]),  // f2(1810)
    polar<float>(params[13], params[14]),  // f2(1950)
    polar<float>(params[15], params[16]),  // a0(980)
    polar<float>(params[17], params[18]),  // a0(1450)
    polar<float>(params[19], params[20]),  // a2(1320)
    polar<float>(params[21], params[22]),  // a2(1700)
  };
  arma::cx_fmat bw_f0 = arma::cx_fmat(5, 5, arma::fill::ones);
  arma::cx_fmat bw_a0 = arma::cx_fmat(2, 2, arma::fill::ones);
  float log_likelihood = 0.0;
  cout << "Calculating data (" << data.nEvents << " events)" << endl;
  for (size_t i = 0; i < data.nEvents; i++) {
    printLoadingBar(i, data.nEvents);
    try {
      log_likelihood += data.weights[i]
        * log(
            amplitude.intensity(
              betas,
              pow(data.masses[i], 2),
              data.thetas[i],
              data.phis[i],
              bw_f0,
              bw_f2[i],
              bw_a0,
              bw_a2[i],
              ikc_inv_vec_f0[i],
              ikc_inv_vec_f2[i],
              ikc_inv_vec_a0[i],
              ikc_inv_vec_a2[i]
              )
            );
    } catch (...) {
      cout << "Singular matrix (event " << i <<") -> skipping..." << endl << endl;
    }
  }
  cout << "Calculating MC (" << acc.nEvents << " events)" << endl;
  for (size_t i = 0; i < acc.nEvents; i++) {
    printLoadingBar(i, acc.nEvents);
    try {
      log_likelihood -= acc.weights[i] * amplitude.intensity(
          betas,
          pow(acc.masses[i], 2),
          acc.thetas[i],
          acc.phis[i],
          bw_f0,
          bw_f2_mc[i],
          bw_a0,
          bw_a2_mc[i],
          ikc_inv_vec_f0_mc[i],
          ikc_inv_vec_f2_mc[i],
          ikc_inv_vec_a0_mc[i],
          ikc_inv_vec_a2_mc[i]
          ) / nGenerated;
    } catch (...) {
      cout << "Singular matrix (event " << i <<") -> skipping..." << endl << endl;
    }
  }
  return log_likelihood;
}


void Likelihood::printLoadingBar (
    const int& progress,
    const int& total,
    const int& barWidth) const {
    float ratio = static_cast<float>(progress) / total;
    int completedWidth = static_cast<int>(ratio * barWidth);

    std::cout << "Progress: [";
    for (int i = 0; i < completedWidth; ++i) {
        std::cout << "=";
    }
    for (int i = completedWidth; i < barWidth; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(ratio * 100.0) << "%\r";
    std::cout.flush();
}
