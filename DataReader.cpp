#include "DataReader.h"

DataReader::DataReader(const string& filePath, const string& treeName) {
  // Open the ROOT file in read-only mode
  file = TFile::Open(filePath.c_str(), "READ");
  // Access the TTree from the ROOT file
  tree = dynamic_cast<TTree*>(file->Get(treeName.c_str()));

  // Check if the TTree exists
  if (!tree) {
    cout << "Error: TTree '" << treeName << "' not found!" << endl;
    return;
  }
}

DataReader::~DataReader() {
  // Close the ROOT file
  if (file)
    file->Close();
}

void DataReader::read() {
  // Variables to hold branch values
  double s, theta, phi;

  // Set branch addresses
  tree->SetBranchAddress("s", &s);
  tree->SetBranchAddress("theta", &theta);
  tree->SetBranchAddress("phi", &phi);

  // Loop over the tree entries
  Long64_t numEntries = tree->GetEntries();
  for (Long64_t entry = 0; entry < numEntries; entry++) {
    tree->GetEntry(entry);

    // Perform the calculation or store the values
    // Example: storing values in vectors
    sValues.push_back(s);
    thetaValues.push_back(theta);
    phiValues.push_back(phi);
  }
}

vector<double> DataReader::get_s_vec() const {
  return sValues;
}

vector<double> DataReader::get_theta_vec() const {
  return thetaValues;
}

vector<double> DataReader::get_phi_vec() const {
  return phiValues;
}

int DataReader::getTotalEvents() const {
  if (tree)
    return tree->GetEntries();
  else
    return 0;
}
