#ifndef DATAREADER_H
#define DATAREADER_H

#include <iostream>
#include <string>
#include <vector>
#include <TFile.h>
#include <TTree.h>

using namespace std;

class DataReader {
public:
  DataReader(const string& filePath, const string& treeName);
  ~DataReader();

  void read();

  vector<double> get_s_vec() const;
  vector<double> get_theta_vec() const;
  vector<double> get_phi_vec() const;

  int getTotalEvents() const;

private:
  TFile* file;
  TTree* tree;
  vector<double> sValues;
  vector<double> thetaValues;
  vector<double> phiValues;
};

#endif
