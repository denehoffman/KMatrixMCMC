#ifndef DATAREADER_H
#define DATAREADER_H
#pragma once

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

  int nEvents;
  vector<float> masses;
  vector<float> weights;
  vector<float> thetas;
  vector<float> phis;

private:
  TFile* file;
  TTree* tree;
};

#endif  // DATAREADER_H
