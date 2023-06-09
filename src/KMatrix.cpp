#include "KMatrix.hpp"

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
      stringstream error;
      error << "Error: J = " << J << " is not supported!";
      throw runtime_error(error.str());
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
    const arma::fmat& m_alphas,
    const arma::fmat& m_channels,
    const arma::fmat& g_alphas,
    const arma::fmat& c_bkg) {

  if (m_alphas.n_rows != 1 || m_alphas.n_cols != numAlphas) {
    stringstream error;
    error << "Error: Invalid dimensions for nAlphas matrix: " << m_alphas.n_rows << "x" << m_alphas.n_cols;
    throw runtime_error(error.str());
  }

  if (m_channels.n_rows != numChannels || m_channels.n_cols != 2) {
    stringstream error;
    error << "Error: Invalid dimensions for nChannels matrix: " << m_channels.n_rows << "x" << m_channels.n_cols;
    throw runtime_error(error.str());
  }

  if (g_alphas.n_rows != numChannels || g_alphas.n_cols != numAlphas) {
    stringstream error;
    error << "Error: Invalid dimensions for gAlphas matrix: " << g_alphas.n_rows << "x" << g_alphas.n_cols;
    throw runtime_error(error.str());
  }

  if (c_bkg.n_rows != numChannels || c_bkg.n_cols != numChannels) {
    stringstream error;
    error << "Error: Invalid dimensions for cBkg matrix: " << c_bkg.n_rows << "x" << c_bkg.n_cols;
    throw runtime_error(error.str());
  }

  mAlphas = arma::conv_to<arma::cx_fmat>::from(m_alphas);
  mChannels = arma::conv_to<arma::cx_fmat>::from(m_channels);
  m1s = mChannels.col(0);
  m2s = mChannels.col(1);
  gAlphas = arma::conv_to<arma::cx_fmat>::from(g_alphas);
  cBkg = arma::conv_to<arma::cx_fmat>::from(c_bkg);

  if (J == 0) {
    bwAlphaMat = arma::fmat(numChannels, numAlphas, arma::fill::ones);
    bwAlphaCube = arma::fcube(numChannels, numChannels, numAlphas, arma::fill::ones);
  } else if (J == 2) {
    arma::cx_fmat qAlphas = arma::cx_fmat(numChannels, numAlphas, arma::fill::zeros);
    arma::cx_fmat sAlphas = arma::square(mAlphas);
    arma::cx_fmat m1sRep = arma::repmat(m1s, 1, numAlphas);
    arma::cx_fmat m2sRep = arma::repmat(m2s, 1, numAlphas);
    arma::cx_fmat sAlphasRep = arma::repmat(sAlphas, numChannels, 1);
    qAlphas += arma::sqrt((arma::square(m1sRep + m2sRep) - sAlphasRep) % (arma::square(m1sRep - m2sRep) - sAlphasRep) / (4.0 * sAlphasRep));
    arma::fmat z = arma::real(arma::square(qAlphas) / (0.1973 * 0.1973));
    bwAlphaMat = arma::sqrt(13.0 * arma::square(z) / (arma::square(z - 3.0) + 9.0 * z));
    bwAlphaCube = arma::fcube(numChannels, numChannels, numAlphas, arma::fill::zeros);
    for (size_t k = 0; k < numAlphas; k++) {
      arma::fvec bwVec = bwAlphaMat.col(k);
      bwAlphaCube.slice(k) = bwVec * bwVec.t();
    }
  } else {

  }
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
arma::cx_fvec KMatrix::chi_p(const float& s) const {
  arma::cx_fvec result(numChannels, arma::fill::ones);
  result -= arma::square(m1s + m2s) / s;
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
arma::cx_fvec KMatrix::chi_m(const float& s) const {
  arma::cx_fvec result(numChannels, arma::fill::ones);
  result -= arma::square(m1s - m2s) / s;
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
arma::cx_fvec KMatrix::rho(const float& s) const {
  arma::cx_fvec result(numChannels, arma::fill::zeros);
  result += arma::sqrt((arma::square(m1s + m2s) - s) % (arma::square(m1s - m2s) - s) / (s * s)); // TODO: check this
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
arma::cx_fvec KMatrix::q(const float& s) const {
  arma::cx_fvec result(numChannels, arma::fill::zeros);
  result += arma::sqrt((arma::square(m1s + m2s) - s) % (arma::square(m1s - m2s) - s) / (4.0 * s)); // TODO: check this
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
arma::fvec KMatrix::blatt_weisskopf0(const float& s) {
  return arma::fvec(numChannels, arma::fill::ones);
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
arma::fvec KMatrix::blatt_weisskopf2(const float& s) {
  arma::fvec z = arma::real(arma::square(KMatrix::q(s)) / (0.1973 * 0.1973));
  arma::fvec result = arma::sqrt(13.0 * arma::square(z) / (arma::square(z - 3.0) + 9.0 * z));
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
arma::fvec KMatrix::blatt_weisskopf(const float& s) const {
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
arma::fmat KMatrix::B(const float& s) const {
  arma::fmat result(numChannels, numAlphas, arma::fill::zeros);
  result.each_col() += KMatrix::blatt_weisskopf(s);
  result /= bwAlphaMat;
  // for (size_t j = 0; j < numAlphas; j++) {
  //   result.col(j) /= KMatrix::blatt_weisskopf((mAlphas(j) * mAlphas(j)).real());
  // }
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
arma::fcube KMatrix::B2(const float& s) const {
  arma::fcube result(numChannels, numChannels, numAlphas, arma::fill::zeros);
  arma::fvec numerator = KMatrix::blatt_weisskopf(s);
  result.each_slice() += numerator * numerator.st();
  result /= bwAlphaCube;
  // for (size_t k = 0; k < numAlphas; k++) {
  //   result.slice(k) /= KMatrix::blatt_weisskopf((mAlphas(k) * mAlphas(k)).real()) * KMatrix::blatt_weisskopf((mAlphas(k) * mAlphas(k)).real()).t();
  // }
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
arma::cx_fmat KMatrix::K(const float& s) const {
  arma::cx_fmat result(numChannels, numChannels, arma::fill::zeros);
  arma::cx_fcube gigj = arma::zeros<arma::cx_fcube>(numChannels, numChannels, numAlphas);
  for (size_t k = 0; k < numAlphas; k++) {
    gigj.slice(k) += gAlphas.col(k) * gAlphas.col(k).st();
    gigj.slice(k) /= (s - (mAlphas(k) * mAlphas(k)));
    gigj.slice(k) = gigj.slice(k) + cBkg;
  }
  gigj %= arma::conv_to<arma::cx_fcube>::from(B2(s));
  result += arma::sum(gigj, 2);
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
arma::cx_fmat KMatrix::K(const float& s, const float& s_0, const float& s_norm) const {
  arma::cx_fmat result(numChannels, numChannels, arma::fill::zeros);
  arma::cx_fcube gigj = arma::zeros<arma::cx_fcube>(numChannels, numChannels, numAlphas);
  for (size_t k = 0; k < numAlphas; k++) {
    gigj.slice(k) = gAlphas.col(k) * gAlphas.col(k).st();
    gigj.slice(k) /= (s - mAlphas(k) * mAlphas(k));
    gigj.slice(k) = gigj.slice(k) + cBkg;
  }
  gigj %= arma::conv_to<arma::cx_fcube>::from(B2(s));
  result += arma::sum(gigj, 2);
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
arma::cx_fmat KMatrix::C(const float& s) const {
  arma::cx_fmat result(numChannels, numChannels, arma::fill::zeros);
  arma::cx_fvec diagonal(numChannels, arma::fill::zeros);
  // diagonal += KMatrix::rho(s)
  //   % arma::log((KMatrix::chi_p(s) + KMatrix::rho(s)) / (KMatrix::chi_p(s) - KMatrix::rho(s)) 
  //       + complex<float>(0, +0.0) // this stupid code ensures we have +0i rather than -0i
  //       );
  // diagonal -= KMatrix::chi_p(s)
  //   % ((mChannels.col(1) - mChannels.col(0)) / (mChannels.col(0) + mChannels.col(1)))
  //   % arma::log(mChannels.col(1) / mChannels.col(0));
  // diagonal /= arma::datum::pi;
  const arma::cx_fvec& m1 = mChannels.col(0);
  const arma::cx_fvec& m2 = mChannels.col(0);
  arma::cx_fvec chi_p = KMatrix::chi_p(s);
  arma::cx_fvec rho = KMatrix::rho(s);
  diagonal += rho % arma::log((chi_p + rho) / (chi_p - rho)) + complex<float>(0, +0.0);
  diagonal -= chi_p % ((m2 - m1) / (m1 + m2)) % arma::log(m2 / m1);
  diagonal /= arma::datum::pi;
  result += arma::diagmat(diagonal); // TODO maybe optimize
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
arma::cx_fmat KMatrix::IKC_inv(const float& s) {
  arma::cx_fmat kmat = KMatrix::K(s);
  arma::cx_fmat cmat = KMatrix::C(s);
  arma::cx_fmat IKC = arma::eye<arma::fmat>(numChannels, numChannels) + kmat * cmat;
  arma::cx_fmat result = arma::cx_fmat(numChannels, numChannels, arma::fill::zeros);
  try {
    result = arma::inv(IKC, arma::inv_opts::allow_approx);
  } catch (runtime_error) {
    throw runtime_error("Matrix inverse failed!");
  }
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
arma::cx_fmat KMatrix::IKC_inv(const float& s, const float& s_0, const float& s_norm) {
  arma::cx_fmat kmat = KMatrix::K(s, s_0, s_norm);
  arma::cx_fmat cmat = KMatrix::C(s);
  arma::cx_fmat IKC = arma::eye<arma::fmat>(numChannels, numChannels) + kmat * cmat;
  arma::cx_fmat result = arma::cx_fmat(numChannels, numChannels, arma::fill::zeros);
  try {
    result = arma::inv(IKC, arma::inv_opts::allow_approx);
  } catch (runtime_error) {
    throw runtime_error("Matrix inverse failed!");
  }
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
arma::cx_fvec KMatrix::P(const float& s, const arma::cx_fvec& betas) const {
  arma::cx_fvec result(numChannels, arma::fill::zeros);
  arma::cx_fmat betag(numChannels, numAlphas);
  betag = gAlphas.each_row() % betas.st();
  for (size_t j = 0; j < numAlphas; j++) {
    betag.col(j) /= (s - mAlphas(j) * mAlphas(j));
  }
  betag %= arma::conv_to<arma::cx_fmat>::from(KMatrix::B(s));
  result += arma::sum(betag, 1);
  return result;
}

arma::cx_fvec KMatrix::P(const float& s, const arma::cx_fvec& betas, const arma::fmat& B) const {
  arma::cx_fvec result(numChannels, arma::fill::zeros);
  arma::cx_fmat betag(numChannels, numAlphas);
  betag = gAlphas.each_row() % betas.st();
  for (size_t j = 0; j < numAlphas; j++) {
    betag.col(j) /= (s - mAlphas(j) * mAlphas(j));
  }
  betag %= arma::conv_to<arma::cx_fmat>::from(B);
  result += arma::sum(betag, 1);
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
complex<float> KMatrix::F(const float& s, const arma::cx_fvec& betas, const arma::cx_fvec& ikc_inv_vec) {
  arma::cx_fvec p_vec = KMatrix::P(s, betas);
  return arma::dot(ikc_inv_vec, p_vec);
}

complex<float> KMatrix::F(const float& s, const arma::cx_fvec& betas, const arma::fmat& B, const arma::cx_fvec& ikc_inv_vec) {
  arma::cx_fvec p_vec = KMatrix::P(s, betas, B);
  return arma::dot(ikc_inv_vec, p_vec);
}
