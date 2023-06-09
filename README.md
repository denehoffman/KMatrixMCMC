# KMatrixMCMC

[![Build Status](https://img.shields.io/github/actions/workflow/status/denehoffman/KMatrixMCMC/build.yml)](https://github.com/denehoffman/KMatrixMCMC/actions) ![Top Language](https://img.shields.io/github/languages/top/denehoffman/KMatrixMCMC) ![Total Lines](https://img.shields.io/tokei/lines/github/denehoffman/KMatrixMCMC) ![Code Size](https://img.shields.io/github/languages/code-size/denehoffman/KMatrixMCMC) ![Last Commit](https://img.shields.io/github/last-commit/denehoffman/KMatrixMCMC) ![License](https://img.shields.io/github/license/denehoffman/KMatrixMCMC)

KMatrixMCMC is a C++ project designed to calculate the K-matrix amplitude for the $K_SK_S$ channel of the GlueX dataset and perform an MCMC analysis (Markov-Chain Monte Carlo).

## Table of Contents

- [Introduction](#introduction)
  - [The K-Matrix Amplitude](#the-k-matrix-amplitude)
  - [Constructing a Likelihood Function](#constructing-a-likelihood-function)
  - [Markov-Chain Monte Carlo](#markov-chain-monte-carlo)
- [Installation](#installation)
- [Building the Project](#building-the-project)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Demo](#demo)
- [License](#license)
- [Tasks](#tasks)

## Introduction

The KMatrixMCMC project aims to provide a convenient and efficient way to calculate the K-matrix amplitude for the K_SK_S channel of the GlueX dataset and perform an MCMC analysis. The K-matrix amplitude is an essential component in the study of hadronic scattering processes and is widely used in particle physics research.

### The K-Matrix Amplitude

The K-matrix parameterization[^1] is similar to a sum of Breit-Wigners, but it preserves unitarity for nearby resonances. The complex amplitude for the $i$th final-state channel is written as:
```math
F_i(s;\vec{\beta}) = \sum_j \left( I + K(s) C(s)\right)_{ij}^{-1} \cdot P_j(s;\vec{\beta})
```
where $s$ is the invariant mass squared of $K_SK_S$ for a particular event, $I$ is the identity matrix, $\beta$ is a vector of complex couplings from the initial state to each resonance, and the other matrices and vectors are defined as follows.

First, we define the K-matrix as:
```math
K_{ij}(s) = \left(\frac{s - s_0}{s_\text{norm}}\right)\sum_{\alpha} B^{\ell}\left(q_i(s),q_i(m_\alpha^2)\right) \cdot \left(\frac{g_{\alpha,i}g_{\alpha,j}}{m_\alpha^2 - s} + c_{ij}\right) \cdot B^{\ell}\left(q_j(s),q_j(m_\alpha^2)\right)
```
Here we sum over resonances (labeled $\alpha$) which have a real mass $m_\alpha$ and real coupling $g_{\alpha,i}$ to the $i$th channel. The term in front accounts for the Adler zero in chiral perturbation theory and must be included when handling the $f_0$ K-matrix, as one of the resonances is near the pion mass. Additionally, we can add real terms $c_{ij}$ and preserve unitarity. The outer functions are ratios of Blatt-Weisskopf barrier functions:
```math
B^{\ell}\left(q_i(s),q_i(m_\alpha^2)\right) = \frac{b^{\ell}\left(q_i(s)\right)}{b^{\ell}\left(q_i(m_\alpha^2)\right)}
```
where
```math
b^{\ell}(q) = \begin{cases} 1 & \ell = 0 \\ \sqrt{\frac{13z^2}{(z-3)^2 + 9z}},\ z = \frac{q^2}{q_0^2} & \ell = 2 \end{cases}
```
where $q_0$ is the effective centrifugal barrier momentum, set to $q_0 = 0.1973\text{ GeV}$ in the code. Currently, the barrier factors for $\ell\neq 0, 2$ are not implemented as they are not used in this channel's analysis.

The functions $q_i(s)$ correspond to the breakup momentum of a particle with invariant mass squared of $s$ in the $i$th channel:
```math
q_i(s) = \rho_i(s)\frac{\sqrt{s}}{2}
```
```math
\rho_i(s) = \sqrt{\chi^+_i(s)\chi^-_i(s)}
```
```math
\chi^{\pm}_i(s) = 1 - \frac{(m_{i,1} \pm m_{i,2})^2}{s}
```
where $m_{i,1}$ and $m_{i,2}$ are the masses of the daughter particles in the $i$th channel.

Next, $C(s)$ is the Chew-Mandelstam matrix. This is a diagonal matrix whose diagonal elements are given by the Chew-Mandelstam function[^2]:
```math
\begin{align}
C_{ii}(s) &= C_{ii}(s_{\text{thr}}) - \frac{s - s_{\text{thr}}}{\pi}\int_{s_{\text{thr}}}^{\infty} \text{d}s' \frac{\rho_i(s')}{(s'-s)(s'-s_{\text{thr}})} \\
          &= I(s_{\text{thr}}) + \frac{\rho_i(s)}{\pi}\ln\left[\frac{\chi^+_i(s)+\rho_i(s)}{\chi^+_i(s)-\rho_i(s)}\right] - \frac{\chi^+_i(s)}{\pi}\frac{m_{i,2}-m_{i,1}}{m_{i,1}+m_{i,2}}\ln\frac{m_{i,2}}{m_{i,1}}
\end{align}
```
with $s_{\text{thr}} = (m_{i,1}+m_{i,2})^2$. Additionally, we chose $I(s_{\text{thr}}) = 0$.

The final piece of the amplitude is the P-vector, which has a very similar form to the K-matrix:
```math
P_{j}(s;\vec{\beta}) = \sum_{\alpha} \left(\frac{\beta_{\alpha}g_{\alpha,j}}{m_\alpha^2 - s}\right) \cdot B^{\ell}\left(q_j(s),q_j(m_\alpha^2)\right)
```
where $\beta_\alpha$ is the complex coupling from the initial state to the resonance $\alpha$. In this analysis, the $\beta_\alpha$ factors are the free parameters in the fit. Note that you can add complex terms to the P-vector in much the same way as real terms can be added to the K-matrix without violating unitarity.

In the $K_SK_S$ channel, we have access to resonances with even total angular momentum and isospin $I=0,1$. We label these as $f$ and $a$ particles respectively, and at the energies GlueX accesses, we expect to see several $f_0$, $f_2$, $a_0$, and $a_2$ resonances. Since this project only analyzes one channel, all of the input parameters except for the values of each $\beta_\alpha$ are fixed in the fit according to published results by Kopf et. al[^1]. The spin-2 resonances are additionally multiplied by a factor of $Y_{J=2}^{M=2}\left(\theta_{\text{HX}},\phi_{\text{HX}}\right)$, a D-wave spherical harmonic with moment $M=2$ acting on the spherical angles in the helicity frame.

### Constructing a Likelihood Function

We can form an intensity density function by coherently summing the K-matrices for each type of particle:
```math
\mathcal{I}(s,\Omega;\vec{\beta}) = \lvert F^{f_0}_2(s;\vec{\beta}) + F^{a_0}_1(s;\vec{\beta}) + Y_2^2(\Omega)\left(F^{f_2}_2(s;\vec{\beta}) + F^{a_2}_2(s;\vec{\beta})\right) \rvert^2
```
This function is not normalized and does not account for the detector's acceptance/efficiency, the probability of an event getting detected and making it through all analysis actions. We can, of course, define the normalization as
```math
\mu = \int \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \text{d}s\text{d}\Omega
```
where $\eta(s,\Omega)$ is the detector acceptance as a function of the observables in this analysis. This allows us to write a probability density function (PDF):
```math
\mathcal{P}(s,\Omega;\vec{\beta}) = \frac{1}{\mu}\mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega)
```
The extended maximum likelihood can be written as a product of these PDFs for each event times a Poissonian distribution:
```math
\mathcal{L}(\vec{\beta}) = \frac{e^{-\mu}\mu^N}{N!}\prod_i^N\mathcal{P}(s_i,\Omega_i;\vec{\beta})
```
Since the values for each evaluation of the PDF will be $\in(0,1)$, it is unstable to multiply them. Instead, we take the natural logarithm and minimize the negative log-likelihood:
```math
\begin{align}
\ln\mathcal{L}(\vec{\beta}) &= -\mu + N\ln(\mu) - \ln(N!) + \sum_i^N \ln(\mathcal{P}(s_i,\Omega_i;\vec{\beta})) \\
                            &= \sum_i^N \left[\ln\left(\mathcal{I}(s_i,\Omega_i;\vec{\beta})\right) + \ln\left(\eta(s_i,\Omega_i)\right) - \ln(\mu) \right] -\mu + N\ln(\mu) - \ln(N!) \\
                            &= \sum_i^N \left[\ln\left(\mathcal{I}(s_i,\Omega_i;\vec{\beta})\right) + \ln\left(\eta(s_i,\Omega_i)\right)\right] - N\ln(\mu) -\mu + N\ln(\mu) - \ln(N!) \\
                            &= \sum_i^N \ln\mathcal{I}(s_i,\Omega_i;\vec{\beta}) -\mu + N\ln\eta(s_i,\Omega_i) - \ln(N!) \\
                            &= \sum_i^N \ln\mathcal{I}(s_i,\Omega_i;\vec{\beta}) -\int \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \text{d}s\text{d}\Omega - \ln(N!) + N\ln\eta(s_i,\Omega_i) \\
\end{align}
```
We must now compute this integral, but in general it does not have an analytic form, so we resort to Monte Carlo methods. Using the Mean Value Theorem, we know that integrating a function over a domain $D$ with area $A$ gives us the average value of that function times $A$:
```math
\int_D f(x)\text{d}x = A\langle f(x) \rangle
```
We can therefore use a Monte Carlo sample, letting $\eta(s,\Omega)$ be equal to $1$ for accepted events and $0$ for rejected events, to numerically compute the average:
```math
4\pi(s_{\text{max}} - s_{\text{min}})\langle \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \rangle \approx \frac{4\pi(s_{\text{max}} - s_{\text{min}})}{N_{\text{gen}}}\sum_{i}^{N_{\text{acc}}}\mathcal{I}(s_i,\Omega_i;\vec{\beta}) = \int \mathcal{I}(s,\Omega;\vec{\beta})\eta(s,\Omega) \text{d}s\text{d}\Omega
```
All together, we end up with
```math
-\ln\mathcal{L}(\vec{\beta}) = -\left(\sum_i^N \ln\mathcal{I}(s_i,\Omega_i;\vec{\beta}) - \frac{4\pi(s_{\text{max}} - s_{\text{min}})}{N_{\text{gen}}}\sum_{i}^{N_{\text{acc}}}\mathcal{I}(s_i,\Omega_i;\vec{\beta})\right) - \ln(N!) + N\ln\eta(s_i,\Omega_i)
```
We can, of course, compute $\ln(N!)$, but the final term is still unknown. However, it doesn't depend on the free parameters $\vec{\beta}$, so it vanishes when we minimize with respect to $\vec{\beta}$ (as does $\ln(N!)$, but it's inexpensive to calculate and we can do so if we want to).

### Markov-Chain Monte Carlo

[TODO]

## Installation

To use the KMatrixMCMC program, you need to install the following dependencies:

- CERN ROOT: Please refer to the [CERN ROOT webpage](https://root.cern/install/) for installation instructions specific to your operating system.

- OpenBLAS and LAPACK: You can install these dependencies using your system's native package manager. Here are examples for various operating systems:

  - **MacOS** (using Homebrew):
    ```shell
    brew install openblas lapack
    ```

  - **Ubuntu** or **Debian**:
    ```shell
    sudo apt-get install libopenblas-dev liblapack-dev
    ```

  - **Arch** (using Pacman):
    ```shell
    sudo pacman -S openblas lapack
    ```

- Armadillo: The latest version of Armadillo should be installed from source. Follow these steps:

  1. Download the Armadillo source code using `wget`:
     ```shell
     wget http://sourceforge.net/projects/arma/files/armadillo-12.4.0.tar.xz
     ```

  2. Untar the downloaded file:
     ```shell
     tar -xf armadillo-12.4.0.tar.xz
     ```

  3. Navigate to the extracted directory:
     ```shell
     cd armadillo-12.4.0
     ```

  4. Build and install Armadillo:
     ```shell
     cmake .
     sudo make install
     ```

     Note: You may need to install CMake if you don't have it already (`sudo apt-get install cmake`).

Once you have installed the above dependencies, you can proceed with building and using the KMatrixMCMC program as described in the previous sections.

## Building the Project

To build the project from source, follow these steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/denehoffman/KMatrixMCMC.git
   ```

2. Navigate to the project directory:
   ```shell
   cd KMatrixMCMC
   ```

3. Generate the build files using CMake:
   ```shell
   cmake -B build
   ```

4. Build the project:
   ```shell
   cmake --build build
   ```

5. Optionally, install the executable:
   ```shell
   sudo cmake --install build
   ```
   This installs to the system default (usually `/usr/local/bin` on **MacOS** or **Linux**). Alternatively, you can install to any specified location:
   ```shell
   cmake --install build --prefix /path/to/install/location/
   ```

## Usage

The `kmatrix_mcmc` executable takes three arguments: paths to the data, accepted Monte Carlo, and generated Monte Carlo CERN ROOT files. The files must have the following branches:

- `Weight`: Event weight
- `E_Beam`, `Px_Beam`, `Py_Beam`, `Pz_Beam`: Components of the beam 4-momentum
- `E_FinalState`, `Px_FinalState`, `Py_FinalState`, `Pz_FinalState`: Arrays of three floats constructing 4-momenta of the three final state particles (proton and two kaons)

Example usage:
```shell
kmatrix_mcmc data.root accmc.root genmc.root
```

## Data Requirements

In order to run the `kmatrix_mcmc` executable, the data, accepted Monte Carlo, and generated Monte Carlo CERN ROOT files must adhere to the required format specified above. Make sure your files contain the necessary branches and the appropriate data. The generated file serves only to provide the number of generated events, and the data contained is not actually read into memory.

## Demo

You can test run the executable on some demo files which are located [here](https://cmu.box.com/s/g2jwtxcfhpx0f1hzmcir201ir1yuxb05). This folder contains two ROOT files which contain 10,000 events each (reuse the accepted MC file as the generated for demonstration purposes). After installing this project and downloading the demo files, you can test it by running the following code:
```shell
kmatrix_mcmc data_short.root accmc_short.root accmc_short.root
```

## License

This project is licensed under the [MIT License](LICENSE).

## Tasks

- [ ] Figure out why the code uses so much RAM (it should be 1/10th its current size)
- [ ] Maybe export precomputed K-matrix and Blatt-Weisskopf values to a file on disk instead of just holding them in RAM?
- [ ] Finish documentation
- [ ] Finish unit tests


[^1]: Kopf, B., Albrecht, M., Koch, H., Küßner, M., Pychy, J., Qin, X., & Wiedner, U. Investigation of the lightest hybrid meson candidate with a coupled-channel analysis of $\bar{p}p$-, $\pi^-p$- and $\pi\pi$-Data. *Eur. Phys. J. C* **81**, 1056 (2021). [https://doi.org/10.1140/epjc/s10052-021-09821-2](https://doi.org/10.1140/epjc/s10052-021-09821-2)
[^2]: Wilson, D. J., Dudek, J. J., Edwards, R. G. & Thomas, C. E. Resonances in coupled $\pi K$, $\eta K$ scattering from lattice QCD. *Phys. Rev. D* **91**, 054008 (2015). [https://doi.org/10.1103/PhysRevD.91.054008](https://doi.org/10.1103/PhysRevD.91.054008)
