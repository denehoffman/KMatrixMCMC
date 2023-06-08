#!/usr/bin/env python3
"""Usage:
  create_short_root_file.py <input_file> <n_events>

Arguments:
  <input_file>     Path to the input root file
  <n_events>       Number of events to include in the new root file

Options:
  -h --help        Show this help message
"""
import ROOT
from pathlib import Path
from docopt import docopt

def create_short_root_file(input_file, n_events):
    # Open the input root file
    file = ROOT.TFile.Open(str(input_file), "READ")

    # Get the first tree in the file
    tree = file.GetListOfKeys()[0].ReadObj()

    # Get the total number of entries in the tree
    total_entries = tree.GetEntries()

    # Cap n_events at the total number of entries
    n_events = min(n_events, total_entries)

    # Create the new root file with the "_short" suffix
    output_file = ROOT.TFile(str(input_file.parent / input_file.stem) + "_short.root", "RECREATE")

    # Create the new tree in the output file
    output_tree = tree.CloneTree(0)

    # Loop over the events and fill the new tree
    for i in range(n_events):
        tree.GetEntry(i)
        output_tree.Fill()

    # Write the new tree and branches to the output file
    output_file.Write()
    output_file.Close()

if __name__ == "__main__":
    # Parse the command-line arguments
    arguments = docopt(__doc__)

    # Extract the input file path and number of events
    input_file = Path(arguments["<input_file>"])
    n_events = int(arguments["<n_events>"])

    # Create the short root file
    create_short_root_file(input_file, n_events)
