#!/usr/bin/env python3

"""Usage:
    plot_mass.py <input_file>

Arguments:
    <input_file>    Input ROOT file to analyze

Options:
    -h --help       Show this help message
"""

from docopt import docopt
import ROOT
from pathlib import Path
import array

def analyze_root(input_file):
    # Open the input ROOT file
    input_tree = ROOT.TFile(str(input_file))
    
    # Get the first tree available
    tree = input_tree.GetListOfKeys()[0].ReadObj()
    
    # Create histograms
    hist1 = ROOT.TH1F("hist1", "Invariant Mass of K_{S}K_{S}", 40, 1.0, 2.0)
    hist2 = ROOT.TH1F("hist2", "Invariant Mass of K_{S}K_{S} (Unweighted)", 40, 1.0, 2.0)
    hist3 = ROOT.TH1F("hist3", "Invariant Mass of K_{S}K_{S}", 50, 1.0, 2.0)
    hist4 = ROOT.TH1F("hist4", "Invariant Mass of K_{S}K_{S} (Unweighted)", 50, 1.0, 2.0)
    
    # Set axis labels
    hist1.GetXaxis().SetTitle("IM(K_{S}K_{S}) (GeV/c^{2})")
    hist1.GetYaxis().SetTitle("counts / 25 MeV/c^{2}")
    hist2.GetXaxis().SetTitle("IM(K_{S}K_{S}) (GeV/c^{2})")
    hist2.GetYaxis().SetTitle("counts / 25 MeV/c^{2}")
    hist3.GetXaxis().SetTitle("IM(K_{S}K_{S}) (GeV/c^{2})")
    hist3.GetYaxis().SetTitle("counts / 20 MeV/c^{2}")
    hist4.GetXaxis().SetTitle("IM(K_{S}K_{S}) (GeV/c^{2})")
    hist4.GetYaxis().SetTitle("counts / 20 MeV/c^{2}")
    
    # Read the branches
    M_FinalState = array.array('f', [0])
    Weight = array.array('f', [0])
    tree.SetBranchAddress("M_FinalState", M_FinalState)
    tree.SetBranchAddress("Weight", Weight)
    
    # Loop over the entries
    num_entries = tree.GetEntries()
    for i in range(num_entries):
        tree.GetEntry(i)
        
        # Fill histograms with weighted values
        hist1.Fill(M_FinalState[0], Weight[0])
        hist2.Fill(M_FinalState[0])
        hist3.Fill(M_FinalState[0], Weight[0])
        hist4.Fill(M_FinalState[0])
    
    # Save histograms to the output file
    output_path = Path(input_file)
    output_filename = output_path.stem + "_hist.root"
    output_file = ROOT.TFile(str(output_path.parent / output_filename), "RECREATE")
    hist1.Write()
    hist2.Write()
    hist3.Write()
    hist4.Write()
    output_file.Close()

if __name__ == "__main__":
    arguments = docopt(__doc__)
    input_file = arguments["<input_file>"]
    analyze_root(input_file)
