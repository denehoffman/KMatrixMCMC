# KMatrixMCMC


# Installation

To install, run the following commands from inside this directory:
```sh
cmake -S . -B build
cmake --build build
cmake --install build
```
This will install the `kmatrix_mcmc` executable in the default installation location (usually something like `/usr/local`). To choose a custom installation prefix, run the following instead of the last line shown above:
```sh
cmake --install build --prefix <path/to/installation/directory>
```
This will install the executable in `path/to/installation/directory/bin/kmatrix_mcmc`. Note that with either of these commands, you may need to use `sudo` to give `cmake` the proper copying permissions.
