#include <catch2/catch_all.hpp>
#include <armadillo>
#include "KMatrix.hpp"

TEST_CASE("KMatrix constructor initializes members correctly", "[KMatrix]") {
  SECTION("J = 0") {
    KMatrix<2, 3> kmatrix(0);
    REQUIRE(kmatrix.J == 0);
    REQUIRE(kmatrix.mAlphas.n_rows == 1);
    REQUIRE(kmatrix.mAlphas.n_cols == 3);
    REQUIRE(kmatrix.mChannels.n_rows == 2);
    REQUIRE(kmatrix.mChannels.n_cols == 2);
    REQUIRE(kmatrix.gAlphas.n_rows == 2);
    REQUIRE(kmatrix.gAlphas.n_cols == 3);
    REQUIRE(kmatrix.cBkg.n_rows == 2);
    REQUIRE(kmatrix.cBkg.n_cols == 2);
  }

  SECTION("J = 2") {
    KMatrix<3, 4> kmatrix(2);
    REQUIRE(kmatrix.J == 2);
    REQUIRE(kmatrix.mAlphas.n_rows == 1);
    REQUIRE(kmatrix.mAlphas.n_cols == 4);
    REQUIRE(kmatrix.mChannels.n_rows == 3);
    REQUIRE(kmatrix.mChannels.n_cols == 2);
    REQUIRE(kmatrix.gAlphas.n_rows == 3);
    REQUIRE(kmatrix.gAlphas.n_cols == 4);
    REQUIRE(kmatrix.cBkg.n_rows == 3);
    REQUIRE(kmatrix.cBkg.n_cols == 3);
  }

  // SECTION("Unsupported J") {
  //   REQUIRE_THROWS_AS([]() { 
  //       KMatrix<2, 3> kmatrix(1);
  //       }(), std::runtime_error);
  // }
}

TEST_CASE("KMatrix initialize method", "[KMatrix]") {
  arma::fmat mAlphas(1, 2);
  arma::fmat mChannels(3, 2);
  arma::fmat gAlphas(3, 2);
  arma::fmat cBkg(3, 3);

  // Initialize the matrices with appropriate dimensions
  mAlphas.randu();
  mChannels.randu();
  gAlphas.randu();
  cBkg.randu();

  SECTION("Valid input dimensions") {
    KMatrix<3, 2> kmatrix(0);
    REQUIRE_NOTHROW(kmatrix.initialize(mAlphas, mChannels, gAlphas, cBkg));
  }

  SECTION("Invalid m_alphas dimensions") {
    mAlphas.resize(1, 2 + 1);  // Invalid dimensions
    KMatrix<3, 2> kmatrix(0);
    REQUIRE_THROWS_AS(kmatrix.initialize(mAlphas, mChannels, gAlphas, cBkg), std::runtime_error);
  }

  SECTION("Invalid m_channels dimensions") {
    mChannels.resize(3 + 1, 2);  // Invalid dimensions
    KMatrix<3, 2> kmatrix(0);
    REQUIRE_THROWS_AS(kmatrix.initialize(mAlphas, mChannels, gAlphas, cBkg), std::runtime_error);
  }

  SECTION("Invalid g_alphas dimensions") {
    gAlphas.resize(3, 2 + 1);  // Invalid dimensions
    KMatrix<3, 2> kmatrix(0);
    REQUIRE_THROWS_AS(kmatrix.initialize(mAlphas, mChannels, gAlphas, cBkg), std::runtime_error);
  }

  SECTION("Invalid c_bkg dimensions") {
    cBkg.resize(3 + 1, 3);  // Invalid dimensions
    KMatrix<3, 2> kmatrix(0);
    REQUIRE_THROWS_AS(kmatrix.initialize(mAlphas, mChannels, gAlphas, cBkg), std::runtime_error);
  }
}

TEST_CASE("KMatrix chi_p function", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::cx_fvec expected_result = {
    0.6413304,
    0.2441109,
    0.08144276
  };
  
  arma::cx_fvec result = kmat_a2.chi_p(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix chi_m function", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::cx_fvec expected_result = {
    0.8688693,
    0.9999881,
    0.4792309
  };
  
  arma::cx_fvec result = kmat_a2.chi_m(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix rho function", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::cx_fvec expected_result = {
    0.7464799,
    0.4940728,
    0.1975598
  };
  
  arma::cx_fvec result = kmat_a2.rho(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}


TEST_CASE("KMatrix rho function below threshold", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::cx_fvec expected_result = {
    0.6250123,
    std::complex<float>(0.0, 0.3030483),
    std::complex<float>(0.0, 0.2845612)
  };
  
  arma::cx_fvec result = kmat_a2.rho(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix q function", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::cx_fvec expected_result = {
    0.4255590,
    0.2816649,
    0.1126264
  };
  
  arma::cx_fvec result = kmat_a2.q(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}


TEST_CASE("KMatrix q function below threshold", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::cx_fvec expected_result = {
    0.2964694,
    std::complex<float>(0.0, 0.1437484),
    std::complex<float>(0.0, 0.1349792)
  };
  
  arma::cx_fvec result = kmat_a2.q(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix blatt_weisskopf function (J=0)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(0);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::fvec expected_result = arma::fvec(3, arma::fill::ones);
  
  arma::fvec result = kmat_a2.blatt_weisskopf(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix blatt_weisskopf function below threshold (J=0)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(0);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::fvec expected_result = arma::fvec(3, arma::fill::ones);
  
  arma::fvec result = kmat_a2.blatt_weisskopf(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix blatt_weisskopf function (J=2)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::fvec expected_result = {
    2.511697,
    1.674049,
    0.3699875
  };
  
  arma::fvec result = kmat_a2.blatt_weisskopf(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix blatt_weisskopf function below threshold (J=2)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::fvec expected_result = {
    1.781955,
    0.6902086,
    0.6036541
  };
  
  arma::fvec result = kmat_a2.blatt_weisskopf(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B function (J=0)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(0);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::fmat expected_result = arma::fmat(3, 2, arma::fill::ones);
  
  arma::fmat result = kmat_a2.B(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B function below threshold (J=0)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(0);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::fmat expected_result = arma::fmat(3, 2, arma::fill::ones);
  
  arma::fmat result = kmat_a2.B(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B function (J=2)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::fmat expected_result = {
    {0.8820837, 0.7711838},
    {0.6715552, 0.5238816},
    {0.2296635, 0.1222795}
  };
  
  arma::fmat result = kmat_a2.B(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B function below threshold (J=2)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::fmat expected_result = {
    {0.6258054, 0.5471261},
    {0.2768815, 0.2159958},
    {0.3747081, 0.1995055}
  };
  
  arma::fmat result = kmat_a2.B(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B2 function (J=0)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(0);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::fcube expected_result = arma::fcube(3, 3, 2, arma::fill::ones);
  
  arma::fcube result = kmat_a2.B2(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B2 function below threshold (J=0)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(0);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::fcube expected_result = arma::fcube(3, 3, 2, arma::fill::ones);
  
  arma::fcube result = kmat_a2.B2(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B2 function (J=2)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 1.3;
  
  arma::fcube expected_result = arma::fcube(3, 3, 2, arma::fill::zeros);
  arma::fmat expected_slice_0 = {
      {0.7780716, 0.5923679, 0.2025824},
      {0.5923679, 0.4509864, 0.1542317},
      {0.2025824, 0.1542317, 0.05274532}
  };

  arma::fmat expected_slice_1 = {
      {0.5947244, 0.404009, 0.0943},
      {0.404009, 0.2744519, 0.06406},
      {0.0943, 0.06406, 0.01495229}
  };
  expected_result.slice(0) = expected_slice_0;
  expected_result.slice(1) = expected_slice_1;
  
  arma::fcube result = kmat_a2.B2(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

TEST_CASE("KMatrix B2 function below threshold (J=2)", "[KMatrix]") {
  arma::fmat a2_mchannels = {
    {0.13498, 0.54786},
    {0.49368, 0.49761},
    {0.13498, 0.95778}
  };
  arma::fmat a2_malphas = {1.30080, 1.75351};
  arma::fmat a2_galphas = {
    {+0.30073, +0.21426, -0.09162},
    {+0.68567, +0.12543, +0.00184}
  };
  arma::fmat a2_cbkg = {
    {-0.40184, +0.00033, -0.08707},
    {+0.00033, -0.21416, -0.06193},
    {-0.08707, -0.06193, -0.17435}
  };

  KMatrix<3, 2> kmat_a2(2);
  kmat_a2.initialize(a2_malphas, a2_mchannels, a2_galphas.t(), a2_cbkg);
  
  float s = 0.9;
  
  arma::fcube expected_result = arma::fcube(3, 3, 2, arma::fill::zeros);
  arma::fmat expected_slice_0 = {
      {0.3916324, 0.1732739, 0.2344944},
      {0.1732739, 0.07666336, 0.1037497},
      {0.2344944, 0.1037497, 0.1404062}
  };

  arma::fmat expected_slice_1 = {
      {0.2993469, 0.1181769, 0.1091547},
      {0.1181769, 0.04665419, 0.04309236},
      {0.1091547, 0.04309236, 0.03980245}
  };
  expected_result.slice(0) = expected_slice_0;
  expected_result.slice(1) = expected_slice_1;
  
  arma::fcube result = kmat_a2.B2(s);
  CAPTURE(expected_result);  // Print the expectation
  CAPTURE(result);  // Print the result
  
  REQUIRE(arma::approx_equal(result, expected_result, "reldiff", 0.00001));
}

