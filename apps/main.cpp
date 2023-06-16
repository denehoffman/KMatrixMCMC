#define MCMC_ENABLE_ARMA_WRAPPERS
#define MCMC_FPN_TYPE float
#include "mcmc.hpp"

#include <iostream>
#include <armadillo>
#include <complex>
#include <chrono>
#include <sstream>
#include "KMatrix.hpp"
#include "Amplitude.hpp"
#include "Likelihood.hpp"
#include "DataReader.hpp"
#include <TFile.h>
#include <TGraph.h>
#include <TH1F.h>
#include <TLegend.h>
#include <TCanvas.h>

using namespace std;
using namespace arma;
using namespace mcmc;

float log_target_dens(const fvec& vals_inp, void* ll_data) {
  Likelihood* lh = reinterpret_cast<Likelihood*>(ll_data);
  return lh->getExtendedLogLikelihood(vals_inp);
}
void createPlots(const Cube_t& cube_t_object) {
    size_t n_row = cube_t_object.n_row;
    size_t n_col = cube_t_object.n_col;
    size_t n_mat = cube_t_object.n_mat;

    TFile *file = new TFile("output.root", "RECREATE");

    for (size_t parameter = 0; parameter < n_col; ++parameter) {
        TCanvas *canvas = new TCanvas(Form("canvas_%zu", parameter), "Canvas", 800, 600);

        for (size_t walker = 0; walker < n_row; ++walker) {
            // Create a TGraph for each walker
            TGraph *graph = new TGraph();

            for (size_t step = 0; step < n_mat; ++step) {
                // Retrieve the value from the Cube_t object
                double value = cube_t_object(walker, parameter, step);

                // Add a point to the TGraph
                graph->SetPoint(step, step, value);
            }

            // Set the line color to blue
            graph->SetLineColor(kBlue);

            // Add the TGraph to the canvas
            if (walker == 0) {
                // For the first walker, adjust the y-axis bounds
                graph->Draw("ALP");
                graph->GetYaxis()->SetRangeUser(graph->GetMinimum(), graph->GetMaximum());
            } else {
                // For subsequent walkers, only draw the TGraph
                graph->Draw("LP same");
            }
        }

        // Save the canvas to the TFile
        canvas->Write();

        // Delete the TGraph objects
        canvas->Clear();

        // Delete the canvas to avoid memory leaks
        delete canvas;
    }

    file->Close();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cout << "Insufficient command-line arguments provided." << endl;
    return 1;
  }

  vector<float> params {
    1.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
    1.0, 0.0,
  };
  cout << "Starting Calculation" << endl;
  auto startTime = chrono::high_resolution_clock::now();
  Likelihood lh(argv[1], argv[2], argv[3]);
  auto endTime = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  cout << "Files loaded in " << duration << "ms" << endl;
  startTime = chrono::high_resolution_clock::now();
  lh.setup();
  endTime = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  cout << endl << "Precalculation took " << duration << "ms" << endl;
  startTime = chrono::high_resolution_clock::now();
  cout << lh.getExtendedLogLikelihood(params) << endl;
  endTime = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  cout << endl << "Amplitude calculation took " << duration << "ms" << endl;

  cout << "Attempting MCMC" << endl;
  algo_settings_t settings;
  settings.de_settings.n_burnin_draws = 50;
  settings.de_settings.n_keep_draws = 50;
  settings.lower_bounds = {
    0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0
  };
  settings.upper_bounds = {
    10000.0,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415,
    10000.0, 2 * 3.1415
  };
  settings.vals_bound = true;
  settings.de_settings.initial_lb = settings.lower_bounds;
  settings.de_settings.initial_ub = settings.upper_bounds;
  Cube_t draws_out;
  fvec initial(23);
  startTime = chrono::high_resolution_clock::now();
  mcmc::de(initial, log_target_dens, draws_out, &lh, settings);
  endTime = chrono::high_resolution_clock::now();
  duration = chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  cout << "CHAIN:" << endl << endl << draws_out.mat(settings.de_settings.n_keep_draws - 1) << endl;
  cout << endl << "MCMC took " << duration << "ms" << endl;
  cout << "de mean:\n" << arma::mean(draws_out.mat(settings.de_settings.n_keep_draws - 1)) << endl;
  cout << "acceptance rate: " << static_cast<float>(settings.de_settings.n_accept_draws) / (settings.de_settings.n_keep_draws * settings.de_settings.n_pop) << endl;
  createPlots(draws_out);
  
  return 0;
}
