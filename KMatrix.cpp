#include "KMatrix.h"

//!
//! @brief Constructor for KMatrix class
//!
//! @param[in] numChannels The number of channels in the K-matrix
//! @param[in] numAlphas The number of resonances in the K-matrix
//! @param[in] J The total anglular momentum of all resonances in the K-matrix
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
//! @brief Calculate \f(\chi^+\f) function
//!
//! \f[
//! \chi^+(s) = 1 - \frac{(m_1 + m_2)^2}{s}
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::chi_p(const double& s) const {
  arma::cx_vec result(numChannels);
  result = 1 - arma::square(mChannels.col(0) + mChannels.col(1)) / s;
  return result;
}

//!
//! @brief Calculate \f(\chi^-\f) function
//!
//! \f[
//! \chi^-(s) = 1 - \frac{(m_1 - m_2)^2}{s}
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::chi_m(const double& s) const {
  arma::cx_vec result(numChannels);
  result = 1 - arma::square(mChannels.col(0) - mChannels.col(1)) / s;
  return result;
}

//!
//! @brief Calculate \f(\rho\f) function
//!
//! \f[
//! \rho = \sqrt{\chi^+(s)\chi^-(s)}
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::rho(const double& s) const {
  arma::cx_vec result(numChannels);
  result = arma::sqrt(KMatrix::chi_p(s) % KMatrix::chi_m(s));
  return result;
}

//!
//! @brief Calculate breakup momentum \f(q\f)
//!
//! \f[
//! q = \rho(s) \frac{\sqrt{s}}{2}
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::q(const double& s) const {
  arma::cx_vec result(numChannels);
  result = sqrt(s) * KMatrix::rho(s) / 2.0;
  return result;
}

//!
//! @brief Calculate Blatt-Weisskopf centrifugal barrier factor for \f(J=0\f)
//!
//! \f[
//! bw_0(s) = 1
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::blatt_weisskopf0(const double& s) {
  arma::cx_vec result = arma::ones<arma::cx_vec>(numChannels);
  return result;
}

//!
//! @brief Calculate Blatt-Weisskopf centrifugal barrier factor for \f(J=2\f)
//!
//! \f[
//! bw_2(s) = \sqrt{\frac{13z^2}{(z-3)^2 + 9z}}\quad z = \frac{q(s)^2}{0.1973^2}
//! \f]
//!
//! Note: 0.1973 is in GeV and q has units of GeV, so z is dimensionless
//!
//! @param[in] s Input mass squared
//! \return Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::blatt_weisskopf2(const double& s) {
  arma::cx_vec z = arma::square(KMatrix::q(s)) / (0.1973 * 0.1973);
  arma::cx_vec result = arma::sqrt(13.0 * arma::square(z) / (arma::square(z - 3.0) + 9.0 * z));
  return result;
}

//!
//! @brief Calculate Blatt-Weisskopf centrifugal barrier factor according to \f(J\f) value
//!
//! Uses a function pointer assigned at initialization to choose between \f(J=0\f) or \f(J=2\f)
//!
//! @param[in] s Input mass squared
//! \return Vector containing result of this operation for each channel
//!
arma::cx_vec KMatrix::blatt_weisskopf(const double& s) const {
  return blattWeisskopfPtr(s);
}

//!
//! @brief Calculates ratio of Blatt-Weisskopf centrifugal barrier factors for each channel and resonance
//!
//! \f[
//! B^J_i(s, m_\alpha) = \frac{bw^J_i(s)}{bw^J(m_\alpha^2)}
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Matrix containing result of this operation with dimension (numChannels, numAlphas)
//!
arma::cx_mat KMatrix::B(const double& s) const {
  arma::cx_mat result(numChannels, numAlphas);
  result.each_col() = KMatrix::blatt_weisskopf(s);
  for (size_t j = 0; j < numAlphas; j++) {
    result.col(j) /= KMatrix::blatt_weisskopf((mAlphas(j) * mAlphas(j)).real());
  }
  return result;
}

//!
//! @brief Calculates ratio of Blatt-Weisskopf centrifugal barrier factors for each channel and resonance
//!
//! \f[
//! B2_{ij}^J(s, m_\alpha) = \frac{bw_i^J(s)bw_j^J(s)}{(bw^J(m_\alpha^2))^2}
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Cube containing result of this operation with dimension (numChannels, numChannels, numAlphas)
//!
arma::cx_cube KMatrix::B2(const double& s) const {
  arma::cx_cube result(numChannels, numChannels, numAlphas);
  arma::cx_vec numerator = KMatrix::blatt_weisskopf(s);
  result.each_slice() = numerator * numerator.t();
  for (size_t k = 0; k < numAlphas; k++) {
    result.slice(k) /= KMatrix::blatt_weisskopf((mAlphas(k) * mAlphas(k)).real()) * KMatrix::blatt_weisskopf((mAlphas(k) * mAlphas(k)).real()).t();
  }
  return result;
}

//!
//! @brief Calculates K-Matrix for a given input mass
//!
//! \f[
//! K_{ij}(s) = \sum_\alpha \left(\frac{g_{i,\alpha}g_{j,\alpha}}{m_\alpha^2 - s} + cBkg_{ij}\right) B2_{ij}^J(s, m_\alpha)
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Matrix containing result of this operation with dimension (numChannels, numChannels)
//!
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

//!
//! @brief Calculates K-Matrix for a given input mass with Adler zero term
//!
//! \f[
//! K_{ij}(s) = \frac{s - s_0}{s_{\text{norm}}}\sum_\alpha \left(\frac{g_{i,\alpha}g_{j,\alpha}}{m_\alpha^2 - s} + cBkg_{ij}\right) B2_{ij}^J(s, m_\alpha)
//! \f]
//!
//! @param[in] s Input mass squared
//! @param[in] s_0 Location of Adler zero
//! @param[in] s_norm Normalization factor for Adler zero term
//! \return Matrix containing result of this operation with dimension (numChannels, numChannels)
//!
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

//!
//! @brief Calculates Chew-Mandelstam matrix
//!
//! \f[
//! C(s) = \frac{\rho(s)}{\pi}\ln\left(\frac{\chi^+(s) + \rho(s)}{\chi^+(s) - \rho(s)}\right) - \frac{\chi^+(s)}{\pi}\left(\frac{m_2 - m_1}{m_1 + m_2}\right)\ln\left(\frac{m_2}{m_1}\right)
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Diagonal matrix where each diagonal element is the result of this function for the corresponding channel
//!
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

//!
//! @brief Calculates the inverse of the "IKC" matrix
//!
//! \f[
//! \to (I - K(s)C(s))^{-1}
//! \f]
//!
//! @param[in] s Input mass squared
//! \return Matrix containing the result of this calculation with dimensions of (numChannels, numChannels)
//!
arma::cx_mat KMatrix::IKC_inv(const double& s) {
  arma::cx_mat kmat = KMatrix::K(s);
  arma::cx_mat cmat = KMatrix::C(s);
  arma::cx_mat IKC = arma::eye(numChannels, numChannels) + kmat * cmat;
  arma::cx_mat result = arma::inv(IKC, arma::inv_opts::allow_approx);
  return result;
}

//!
//! @brief Calculates the inverse of the "IKC" matrix with Adler zero term in K-Matrix
//!
//! \f[
//! \to (I - K(s)C(s))^{-1}
//! \f]
//!
//! @param[in] s Input mass squared
//! @param[in] s_0 Location of Adler zero
//! @param[in] s_norm Normalization factor for Adler zero term
//! \return Matrix containing the result of this calculation with dimensions of (numChannels, numChannels)
//!
arma::cx_mat KMatrix::IKC_inv(const double& s, const double& s_0, const double& s_norm) {
  arma::cx_mat kmat = KMatrix::K(s, s_0, s_norm);
  arma::cx_mat cmat = KMatrix::C(s);
  arma::cx_mat IKC = arma::eye(numChannels, numChannels) + kmat * cmat;
  arma::cx_mat result = arma::inv(IKC, arma::inv_opts::allow_approx);
  return result;
}

//!
//! @brief Calculates P-vector for a given input mass and vector of couplings
//!
//! \f[
//! P_{j}(s,\beta) = \sum_\alpha \left(\frac{\beta_\alpha g_{j,\alpha}}{m_\alpha^2 - s}\right) B_{j}^J(s, m_\alpha)
//! \f]
//!
//! @param[in] s Input mass squared
//! @param[in] betas Vector containing complex couplings for each resonance
//! \return Vector containing result of this operation
//!
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

//!
//! @brief Calculates complex amplitude for a given K-Matrix parameterization
//!
//! \f[
//! F(s, \beta) = (I - K(s)C(s))^{-1}\cdot P(s, \beta)
//! \f]
//!
//! Note, the actual calculation just uses one row of \f((I - KC)^{-1}\f) since we don't need to store info for the channels we don't use
//!
//! @param[in] s Input mass squared
//! @param[in] betas Vector containing complex couplings for each resonance
//! @param[in] ikc_inv_vec Vector containing a row of the inverse of the "IKC" matrix for the channel specified at initialization
//!
complex<double> KMatrix::F(const double& s, const arma::cx_vec& betas, const arma::cx_vec& ikc_inv_vec) {
  arma::cx_vec p_vec = KMatrix::P(s, betas);
  return arma::dot(ikc_inv_vec, p_vec);
}
