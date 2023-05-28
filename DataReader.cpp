#include "TLorentzVector.h"
#include "TLorentzRotation.h"
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

  nEvents = tree->GetEntries();
}

DataReader::~DataReader() {
  // Close the ROOT file
  if (file)
    file->Close();
}

void DataReader::read() {
  // Variables to hold branch values
  float weight, e_beam, px_beam, py_beam, pz_beam;
  float e_fs[3], px_fs[3], py_fs[3], pz_fs[3];

  // Set branch addresses
  tree->SetBranchAddress("Weight", &weight);
  tree->SetBranchAddress("E_Beam", &e_beam);
  tree->SetBranchAddress("Px_Beam", &px_beam);
  tree->SetBranchAddress("Py_Beam", &py_beam);
  tree->SetBranchAddress("Pz_Beam", &pz_beam);
  tree->SetBranchAddress("E_FinalState", &e_fs);
  tree->SetBranchAddress("Px_FinalState", &px_fs);
  tree->SetBranchAddress("Py_FinalState", &py_fs);
  tree->SetBranchAddress("Pz_FinalState", &pz_fs);

  // Loop over the tree entries
  Long64_t numEntries = tree->GetEntries();
  for (Long64_t entry = 0; entry < numEntries; entry++) {
    tree->GetEntry(entry);

    TLorentzVector beam(px_beam, py_beam, pz_beam, e_beam);
    TLorentzVector recoil(px_fs[0], py_fs[0], pz_fs[0], e_fs[0]);
    TLorentzVector p1(px_fs[1], py_fs[1], pz_fs[1], e_fs[1]);
    TLorentzVector p2(px_fs[2], py_fs[2], pz_fs[2], e_fs[2]);

    TLorentzVector resonance = p1 + p2;
    TLorentzRotation resRestBoost(-resonance.BoostVector());

    TLorentzVector beam_res = resRestBoost * beam;
    TLorentzVector recoil_res = resRestBoost * recoil;
    TLorentzVector p1_res = resRestBoost * p1;
    
    TVector3 z = -1.0 * recoil_res.Vect().Unit();
    TVector3 y = (beam.Vect().Cross(-recoil.Vect())).Unit();
    TVector3 x = y.Cross(z);

    TVector3 angles(
        p1_res.Vect().Dot(x),
        p1_res.Vect().Dot(y),
        p1_res.Vect().Dot(z)
        );
    // Perform the calculation or store the values
    // Example: storing values in vectors
    masses.push_back(resonance.M());
    thetas.push_back(angles.Theta());
    phis.push_back(angles.Phi());
    weights.push_back(weight);
  }
}
