#define ARMA_USE_BLAS
#define ARMA_USE_LAPACK
#include "Likelihood.hpp"
#include "Amplitude.hpp"
#include "DataReader.hpp"
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
  cout << "Precalculating inverse of (I - KC)" << endl;
  cout << "Data" << endl;
  vector<int> badDataIndices;
  arma::cx_fvec ikc_inv_f0(5, arma::fill::zeros);
  arma::cx_fvec ikc_inv_f2(4, arma::fill::zeros);
  arma::cx_fvec ikc_inv_a0(2, arma::fill::zeros);
  arma::cx_fvec ikc_inv_a2(3, arma::fill::zeros);
  arma::fmat bw_f2_matrix(4, 4, arma::fill::zeros);
  arma::fmat bw_a2_matrix(3, 3, arma::fill::zeros);
  for (int i = 0; i < data.nEvents; i++) {
    try {
      float s = pow(data.masses[i], 2);
      ikc_inv_f0 = amplitude.ikc_inv_vec_f0(s);
      ikc_inv_f2 = amplitude.ikc_inv_vec_f2(s);
      ikc_inv_a0 = amplitude.ikc_inv_vec_a0(s);
      ikc_inv_a2 = amplitude.ikc_inv_vec_a2(s);
      bw_f2_matrix = amplitude.bw_f2(s);
      bw_a2_matrix = amplitude.bw_a2(s);

      ikc_inv_vec_f0.push_back(ikc_inv_f0);
      ikc_inv_vec_f2.push_back(ikc_inv_f2);
      ikc_inv_vec_a0.push_back(ikc_inv_a0);
      ikc_inv_vec_a2.push_back(ikc_inv_a2);
      bw_f2.push_back(bw_f2_matrix);
      bw_a2.push_back(bw_a2_matrix);
    } catch (const runtime_error& e) {
      cout << "One or more matrix inverses failed for event " << i << endl;
      badDataIndices.push_back(i);
    }
  }

  for (auto it = badDataIndices.rbegin(); it != badDataIndices.rend(); it++) {
    data.masses.erase(data.masses.begin() + *it);
    data.weights.erase(data.weights.begin() + *it);
    data.thetas.erase(data.thetas.begin() + *it);
    data.phis.erase(data.phis.begin() + *it);
  }
  cout << "Monte Carlo" << endl;
  vector<int> badMCIndices;
  for (int i = 0; i < acc.nEvents; i++) {
    try {
      float s = pow(acc.masses[i], 2);
      ikc_inv_f0 = amplitude.ikc_inv_vec_f0(s);
      ikc_inv_f2 = amplitude.ikc_inv_vec_f2(s);
      ikc_inv_a0 = amplitude.ikc_inv_vec_a0(s);
      ikc_inv_a2 = amplitude.ikc_inv_vec_a2(s);
      bw_f2_matrix = amplitude.bw_f2(s);
      bw_a2_matrix = amplitude.bw_a2(s);

      ikc_inv_vec_f0_mc.push_back(ikc_inv_f0);
      ikc_inv_vec_f2_mc.push_back(ikc_inv_f2);
      ikc_inv_vec_a0_mc.push_back(ikc_inv_a0);
      ikc_inv_vec_a2_mc.push_back(ikc_inv_a2);
      bw_f2_mc.push_back(bw_f2_matrix);
      bw_a2_mc.push_back(bw_a2_matrix);
    } catch (const runtime_error& e) {
      cout << "One or more matrix inverses failed for event " << i << endl;
      badMCIndices.push_back(i);
    }
  }

  for (auto it = badMCIndices.rbegin(); it != badMCIndices.rend(); it++) {
    acc.masses.erase(acc.masses.begin() + *it);
    acc.weights.erase(acc.weights.begin() + *it);
    acc.thetas.erase(acc.thetas.begin() + *it);
    acc.phis.erase(acc.phis.begin() + *it);
  }
}

float Likelihood::getExtendedLogLikelihood(const arma::Col<float>& params) {
  cx_fvec betas;
  if (params.size() == 23) {
    betas = {
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
  } else if (params.size() == 22) {
    betas = {
      polar<float>(0.0, 0.0),                // f0(500)
      polar<float>(100.0, 0.0),          // f0(980)
      polar<float>(params[0], params[1]),    // f0(1370)
      polar<float>(params[2], params[3]),    // f0(1500)
      polar<float>(params[4], params[5]),    // f0(1710)
      polar<float>(params[6], params[7]),    // f2(1270)
      polar<float>(params[8], params[9]),   // f2(1525)
      polar<float>(params[10], params[11]),  // f2(1810)
      polar<float>(params[12], params[13]),  // f2(1950)
      polar<float>(params[14], params[15]),  // a0(980)
      polar<float>(params[16], params[17]),  // a0(1450)
      polar<float>(params[18], params[19]),  // a2(1320)
      polar<float>(params[20], params[21]),  // a2(1700)
    };
  }
  arma::fmat bw_f0 = arma::fmat(5, 5, arma::fill::ones);
  arma::fmat bw_a0 = arma::fmat(2, 2, arma::fill::ones);
  float log_likelihood = 0.0;
  for (size_t i = 0; i < data.masses.size(); i++) {
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
  }
  for (size_t i = 0; i < acc.masses.size(); i++) {
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
