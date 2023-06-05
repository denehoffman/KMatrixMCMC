#include "KMatrix.h"

//!
//! @brief Constructor for KMatrix class
//!
//! @param[in] numChannels The number of channels in the K-matrix
//! @param[in] numAlphas The number of resonances in the K-matrix
//! @param[in] j The total anglular momentum of all resonances in the K-matrix
//!
KMatrix::KMatrix(int numChannels, int numAlphas, int J) : numAlphas(numAlphas), numChannels(numChannels),
  mAlphas(1, numAlphas), mChannels(numChannels, 2),
  gAlphas(numChannels, numAlphas), cBkg(numChannels, numChannels), J(J) {
    if (J == 0) {
      blattWeisskopfPtr = bind(&KMatrix::blatt_weisskopf0, this, placeholders::_1);
    } else if (J == 2) {
      blattWeisskopfPtr = bind(&KMatrix::blatt_weisskopf2, this, placeholders::_1);
    } else {
      cout << "Error: J = " << J << " is not supported!" << endl;
      return;
    }
  }

//!
//! @brief Initialize components of K-matrix
//!
//! @param[in] m_alphas Array containing masses of each resonance (1 x nAlphas)
//! @param[in] m_channels Array containing pairs of daughter masses (numChannels x 2)
//! @param[in] g_alphas Array containing channel coupings "g" (numChannels x numAlphas)
//! @param[in] c_bkg Array of K-matrix background terms (numChannels x numChannels)
//!
void KMatrix::initialize(
    const arma::mat& m_alphas,
    const arma::mat& m_channels,
    const arma::mat& g_alphas,
    const arma::mat& c_bkg) {
  if (m_alphas.n_rows != 1 || m_alphas.n_cols != numAlphas) {
    cout << "Error: Invalid dimensions for nAlphas matrix: " << arma::size(m_alphas) << endl;
    return;
  }

  if (m_channels.n_rows != numChannels || m_channels.n_cols != 2) {
    cout << "Error: Invalid dimensions for nChannels matrix: " << arma::size(m_channels) << endl;
    return;
  }

  if (g_alphas.n_rows != numChannels || g_alphas.n_cols != numAlphas) {
    cout << "Error: Invalid dimensions for gAlphas matrix: " << arma::size(g_alphas) << endl;
    return;
  }

  if (c_bkg.n_rows != numChannels || c_bkg.n_cols != numChannels) {
    cout << "Error: Invalid dimensions for cBkg matrix: " << arma::size(c_bkg) << endl;
    return;
  }

  mAlphas = arma::conv_to<arma::cx_mat>::from(m_alphas);
  mChannels = arma::conv_to<arma::cx_mat>::from(m_channels);
  gAlphas = arma::conv_to<arma::cx_mat>::from(g_alphas);
  cBkg = arma::conv_to<arma::cx_mat>::from(c_bkg);
}

//!
//! @brief Prints the data contained in the K-matrix
//!
//!
void KMatrix::print() const {
  cout << "mAlphas matrix:\n" << mAlphas << endl;
  cout << "mChannels matrix:\n" << mChannels << endl;
  cout << "gAlphas matrix:\n" << gAlphas << endl;
  cout << "cBkg matrix:\n" << cBkg << endl;
}

//!
//! @brief Calculate \(\Chi^+\) function
//!
//! \[
//! \Chi^+(s) = 1 - \frac{(m_1 + m_2)^2}{s}
//! \]
//!
//! @param[in] s Input mass squared
//! @param[out] result Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::chi_p(const double& s) const {
  arma::cx_vec result(numChannels);
  result = 1 - arma::square(mChannels.col(0) + mChannels.col(1)) / s;
  return result;
}

//!
//! @brief Calculate \(\Chi^-\) function
//!
//! \[
//! \Chi^-(s) = 1 - \frac{(m_1 - m_2)^2}{s}
//! \]
//!
//! @param[in] s Input mass squared
//! @param[out] result Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::chi_m(const double& s) const {
  arma::cx_vec result(numChannels);
  result = 1 - arma::square(mChannels.col(0) - mChannels.col(1)) / s;
  return result;
}

//!
//! @brief Calculate \(\rho\) function
//!
//! \[
//! \rho = \sqrt{\Chi^+(s)\Chi^-(s)}
//! \]
//!
//! @param[in] s Input mass squared
//! @param[out] result Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::rho(const double& s) const {
  arma::cx_vec result(numChannels);
  result = arma::sqrt(KMatrix::chi_p(s) % KMatrix::chi_m(s));
  return result;
}

arma::cx_vec KMatrix::q(const double& s) const {
  arma::cx_vec result(numChannels);
  result = sqrt(s) * KMatrix::rho(s) / 2.0;
  return result;
}

// arma::cx_vec KMatrix::blatt_weisskopf(const double& s) const {
//   arma::cx_vec result = arma::ones<arma::cx_vec>(numChannels) * (J == 0);
//   arma::cx_vec z = arma::square(KMatrix::q(s)) / (0.1973 * 0.1973);
//   // result += arma::sqrt(z / (z + 1.0)) * (J == 1);
//   result += arma::sqrt(13.0 * arma::square(z) / (arma::square(z - 3.0) + 9.0 * z)) * (J == 2);
//   // result += arma::sqrt(277.0 * arma::pow(z, 3) / (z % arma::square(z - 15.0) + 9.0 * arma::square(2.0 * z - 5.0))) * (J == 3);
//   // result += arma::sqrt(12746.0 * arma::pow(z, 4) / (arma::square(arma::square(z) - 45.0 * z + 105.0) + 25.0 * z % arma::square(2.0 * z - 21.0))) * (J == 4);
//   return result;
// }

arma::cx_vec KMatrix::blatt_weisskopf0(const double& s) {
  arma::cx_vec result = arma::ones<arma::cx_vec>(numChannels) * (J == 0);
  return result;
}

arma::cx_vec KMatrix::blatt_weisskopf2(const double& s) {
  arma::cx_vec z = arma::square(KMatrix::q(s)) / (0.1973 * 0.1973);
  arma::cx_vec result = arma::sqrt(13.0 * arma::square(z) / (arma::square(z - 3.0) + 9.0 * z)) * (J == 2);
  return result;
}

arma::cx_vec KMatrix::blatt_weisskopf(const double& s) const {
  return blattWeisskopfPtr(s);
}

arma::cx_mat KMatrix::B(const double& s) const {
  arma::cx_mat result(numChannels, numAlphas);
  result.each_col() = KMatrix::blatt_weisskopf(s);
  for (size_t j = 0; j < numAlphas; j++) {
    result.col(j) /= KMatrix::blatt_weisskopf((mAlphas(j) * mAlphas(j)).real());
  }
  return result;
}

arma::cx_cube KMatrix::B2(const double& s) const {
  arma::cx_cube result(numChannels, numChannels, numAlphas);
  arma::cx_vec numerator = KMatrix::blatt_weisskopf(s);
  result.each_slice() = numerator * numerator.t();
  for (size_t k = 0; k < numAlphas; k++) {
    result.slice(k) /= KMatrix::blatt_weisskopf((mAlphas(k) * mAlphas(k)).real()) * KMatrix::blatt_weisskopf((mAlphas(k) * mAlphas(k)).real()).t();
  }
  return result;
}

arma::cx_mat KMatrix::K(const double& s) const {
  arma::cx_mat result(numChannels, numChannels);
  arma::cx_cube gigj = arma::zeros<arma::cx_cube>(numChannels, numChannels, numAlphas);
  for (size_t k = 0; k < numAlphas; k++) {
    gigj.slice(k) += gAlphas.col(k) * gAlphas.col(k).t();
    gigj.slice(k) /= (s - mAlphas(k) * mAlphas(k));
    gigj.slice(k) = gigj.slice(k) + cBkg;
  }
  gigj %= B2(s);
  result = arma::sum(gigj, 2);
  return result;
}

arma::cx_mat KMatrix::K(const double& s, const double& s_0, const double& s_norm) const {
  arma::cx_mat result(numChannels, numChannels);
  arma::cx_cube gigj = arma::zeros<arma::cx_cube>(numChannels, numChannels, numAlphas);
  for (size_t k = 0; k < numAlphas; k++) {
    gigj.slice(k) = gAlphas.col(k) * gAlphas.col(k).t();
    gigj.slice(k) /= (s - mAlphas(k) * mAlphas(k));
    gigj.slice(k) = gigj.slice(k) + cBkg;
  }
  gigj %= B2(s);
  result = arma::sum(gigj, 2);
  return result * (s - s_0) / s_norm;
}

arma::cx_mat KMatrix::C(const double& s) const {
  arma::cx_mat result(numChannels, numChannels);
  arma::cx_vec diagonal(numChannels);
  diagonal += KMatrix::rho(s)
    % arma::log((KMatrix::chi_p(s) + KMatrix::rho(s)) / (KMatrix::chi_p(s) - KMatrix::rho(s)) 
        + complex<double>(0, +0.0) // this stupid code ensures we have +0i rather than -0i
        );
  diagonal -= KMatrix::chi_p(s)
    % ((mChannels.col(1) - mChannels.col(0)) / (mChannels.col(0) + mChannels.col(1)))
    % arma::log(mChannels.col(1) / mChannels.col(0));
  diagonal /= arma::datum::pi;
  result = arma::diagmat(diagonal);
  return result;
}

arma::cx_mat KMatrix::IKC_inv(const double& s) {
  arma::cx_mat kmat = KMatrix::K(s);
  arma::cx_mat cmat = KMatrix::C(s);
  arma::cx_mat IKC = arma::eye(numChannels, numChannels) + kmat * cmat;
  arma::cx_mat result = arma::inv(IKC, arma::inv_opts::allow_approx);
  return result;
}

arma::cx_mat KMatrix::IKC_inv(const double& s, const double& s_0, const double& s_norm) {
  arma::cx_mat kmat = KMatrix::K(s, s_0, s_norm);
  arma::cx_mat cmat = KMatrix::C(s);
  arma::cx_mat IKC = arma::eye(numChannels, numChannels) + kmat * cmat;
  arma::cx_mat result = arma::inv(IKC, arma::inv_opts::allow_approx);
  return result;
}

arma::cx_vec KMatrix::P(const double& s, const arma::cx_vec& betas) const {
  arma::cx_vec result(numChannels);
  arma::cx_mat betag(numChannels, numAlphas);
  betag = gAlphas.each_row() % betas.t();
  for (size_t j = 0; j < numAlphas; j++) {
    betag.col(j) /= (s - mAlphas(j) * mAlphas(j));
  }
  betag %= KMatrix::B(s);
  result = arma::sum(betag, 1);
  return result;
}

complex<double> KMatrix::F(const double& s, const arma::cx_vec& betas, const arma::cx_vec& ikc_inv_vec) {
  arma::cx_vec p_vec = KMatrix::P(s, betas);
  return arma::dot(ikc_inv_vec, p_vec);
}
