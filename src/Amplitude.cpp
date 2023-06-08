#include "Amplitude.hpp"
#include "KMatrix.hpp"

Amplitude::Amplitude() {
  f0_mchannels = {
    {0.13498, 0.13498},
    {0.26995, 0.26995},
    {0.49368, 0.49761},
    {0.54786, 0.54786},
    {0.54786, 0.95778}
  };
  f0_malphas = {0.51461, 0.90630, 1.23089, 1.46104, 1.69611};
  f0_galphas = {
    {+0.74987, -0.01257, +0.02736, -0.15102, +0.36103},
    {+0.06401, +0.00204, +0.77413, +0.50999, +0.13112},
    {-0.23417, -0.01032, +0.72283, +0.11934, +0.36792},
    {+0.01570, +0.26700, +0.09214, +0.02742, -0.04025},
    {-0.14242, +0.22780, +0.15981, +0.16272, -0.17397}
  };
  f0_cbkg = {
    {+0.03728, +0.00000, -0.01398, -0.02203, +0.01397},
    {+0.00000, +0.00000, +0.00000, +0.00000, +0.00000},
    {-0.01398, +0.00000, +0.02349, +0.03101, -0.04003},
    {-0.02203, +0.00000, +0.03101, -0.13769, -0.06722},
    {+0.01397, +0.00000, -0.04003, -0.06722, -0.28401}
  };
  kmat_f0.initialize(f0_malphas, f0_mchannels, f0_galphas.t(), f0_cbkg);

  f2_mchannels = {
    {0.13498, 0.13498},
    {0.26995, 0.26995},
    {0.49368, 0.49761},
    {0.54786, 0.54786}
  };
  f2_malphas = {1.15299, 1.48359, 1.72923, 1.96700};
  f2_galphas = {
    {+0.40033, +0.15479, -0.08900, -0.00113},
    {+0.01820, +0.17300, +0.32393, +0.15256},
    {-0.06709, +0.22941, -0.43133, +0.23721},
    {-0.49924, +0.19295, +0.27975, -0.03987}
  };
  f2_cbkg = {
    {-0.04319, +0.00000, +0.00984, +0.01028},
    {+0.00000, +0.00000, +0.00000, +0.00000},
    {+0.00984, +0.00000, -0.07344, +0.05533},
    {+0.01028, +0.00000, +0.05533, -0.05183}
  };
  kmat_f2.initialize(f2_malphas, f2_mchannels, f2_galphas.t(), f2_cbkg);

  a0_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761}
  };
  a0_malphas = {0.95395, 1.26767};
  a0_galphas = {
    {+0.43215, -0.28825},
    {+0.19000, +0.43372}
  };
  a0_cbkg = {
    {+0.00000, +0.00000},
    {+0.00000, +0.00000},
  };
  kmat_a0.initialize(a0_malphas, a0_mchannels, a0_galphas.t(), a0_cbkg);

  a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  a2_malphas = {1.30080, 1.75351};
  a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
}

cx_fvec Amplitude::ikc_inv_vec_f0(const float& s) {
  try {
    return kmat_f0.IKC_inv(s, 0.0091125, 1.0).col(2);
  } catch (runtime_error) {
    throw runtime_error("Matrix inverse failed!");
  }
}
cx_fvec Amplitude::ikc_inv_vec_f2(const float& s) {
  try {
    return kmat_f2.IKC_inv(s).col(2);
  } catch (runtime_error) {
    throw runtime_error("Matrix inverse failed!");
  }
}
cx_fvec Amplitude::ikc_inv_vec_a0(const float& s) {
  try {
    return kmat_a0.IKC_inv(s).col(1);
  } catch (runtime_error) {
    throw runtime_error("Matrix inverse failed!");
  }
}
cx_fvec Amplitude::ikc_inv_vec_a2(const float& s) {
  try {
    return kmat_a2.IKC_inv(s).col(1);
  } catch (runtime_error) {
    throw runtime_error("Matrix inverse failed!");
  }
}

cx_fmat Amplitude::bw_f0(const float& s) {
  cx_fmat res = kmat_f0.B(s);
  return res;
}
cx_fmat Amplitude::bw_f2(const float& s) {
  cx_fmat res = kmat_f2.B(s);
  return res;
}
cx_fmat Amplitude::bw_a0(const float& s) {
  cx_fmat res = kmat_a0.B(s);
  return res;
}
cx_fmat Amplitude::bw_a2(const float& s) {
  cx_fmat res = kmat_a2.B(s);
  return res;
}

complex<float> Amplitude::S0_wave(const float& theta, const float& phi) {
  return complex<float>(sqrt(1.0 / datum::pi) / 2.0, 0.0);
}

complex<float> Amplitude::D2_wave(const float& theta, const float& phi) {
  return static_cast<float>(pow(sin(theta), 2)) * exp(complex<float>(0.0, 2.0 * phi)) * static_cast<float>(sqrt(15.0 / datum::pi / 2.0)) / complex<float>(4.0, 0.0);
}

float Amplitude::intensity(
    const cx_fvec& betas,
    const float& s,
    const float& theta,
    const float& phi,
    const cx_fvec& ikc_inv_vec_f0,
    const cx_fvec& ikc_inv_vec_f2,
    const cx_fvec& ikc_inv_vec_a0,
    const cx_fvec& ikc_inv_vec_a2) {
  complex<float> f_f0 = kmat_f0.F(s, betas.subvec(0, 4), ikc_inv_vec_f0);
  complex<float> f_f2 = kmat_f2.F(s, betas.subvec(5, 8), ikc_inv_vec_f2);
  complex<float> f_a0 = kmat_a0.F(s, betas.subvec(9, 10), ikc_inv_vec_a0);
  complex<float> f_a2 = kmat_a2.F(s, betas.subvec(11, 12), ikc_inv_vec_a2);
  complex<float> S0 = Amplitude::S0_wave(theta, phi);
  complex<float> D2 = Amplitude::D2_wave(theta, phi);
  return pow(abs(S0 * (f_f0 + f_a0) + D2 * (f_f2 + f_a2)), 2);
}

float Amplitude::intensity(
    const cx_fvec& betas,
    const float& s,
    const float& theta,
    const float& phi,
    const cx_fmat& bw_f0,
    const cx_fmat& bw_f2,
    const cx_fmat& bw_a0,
    const cx_fmat& bw_a2,
    const cx_fvec& ikc_inv_vec_f0,
    const cx_fvec& ikc_inv_vec_f2,
    const cx_fvec& ikc_inv_vec_a0,
    const cx_fvec& ikc_inv_vec_a2) {
  complex<float> f_f0 = kmat_f0.F(s, betas.subvec(0, 4), bw_f0, ikc_inv_vec_f0);
  complex<float> f_f2 = kmat_f2.F(s, betas.subvec(5, 8), bw_f2, ikc_inv_vec_f2);
  complex<float> f_a0 = kmat_a0.F(s, betas.subvec(9, 10), bw_a0, ikc_inv_vec_a0);
  complex<float> f_a2 = kmat_a2.F(s, betas.subvec(11, 12), bw_a2, ikc_inv_vec_a2);
  complex<float> S0 = Amplitude::S0_wave(theta, phi);
  complex<float> D2 = Amplitude::D2_wave(theta, phi);
  return pow(abs(S0 * (f_f0 + f_a0) + D2 * (f_f2 + f_a2)), 2);
}
