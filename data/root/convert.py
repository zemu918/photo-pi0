#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: convert.py
description: Convert GEANT4 root files into hdf5 files
author: Luke de Oliveira (lukedeo@manifold.ai)
"""

import os
import ROOT 
from ROOT import * 

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


def root_open(filename, mode=''):
    mode_map = {'a': 'UPDATE',
                'a+': 'UPDATE',
                'r': 'READ',
                'r+': 'UPDATE',
                'w': 'RECREATE',
                'w+': 'RECREATE'}

    if mode in mode_map:
        mode = mode_map[mode]

    filename = expand(filename)
    prev_dir = ROOT.gDirectory
    root_file = ROOT.TFile.Open(filename, mode)
    if not root_file:
        raise IOError("could not open file: '{0}'".format(filename))
#    root_file.__class__ = File
    root_file._path = filename
    root_file._parent = root_file
    root_file._prev_dir = prev_dir
    # give Python ownership of the TFile so we can delete it
    ROOT.SetOwnership(root_file, True)
    return root_file

import root_numpy
#from rootpy.io import root_open
from root_numpy import tree2array
import pandas as pd
import numpy as np
from h5py import File as HDF5File

#LAYER_SPECS = [(3, 96), (12, 12), (12, 6)]
LAYER_SPECS = [(6, 96), (44, 120), (6, 96)]

LAYER_DIV = np.cumsum(map(np.prod, LAYER_SPECS)).tolist()
           #[ 576, 5856, 6432]
LAYER_DIV = zip([0] + LAYER_DIV, LAYER_DIV)
           #[ (0,576), (576,5856), (5856,6432)]

#OVERFLOW_BINS = 4 


def write_out_file(infile, outfile, tree=None):
    f = root_open(infile)
    T = f.Get(tree)

    branchnames = [branch.GetName() for branch in T.GetListOfBranches()]
    print("The branchnames are: ", branchnames) 
    cells = filter(lambda x: x.startswith('cell'), branchnames)
            # collect all branchs begin with cell in a list
    print("The cells are: ", cells) 

    X = pd.DataFrame(tree2array(T, branches=cells)).values
        # X storage all cell-branchs value
   # E = pd.DataFrame(tree2array(T, branches=['TotalEnergy'])).values.ravel()
    print("X.shape",X.shape)
   # print("E.shape",E.shape)

    with HDF5File(outfile, 'w') as h5:
        for layer, (sh, (l, u)) in enumerate(zip(LAYER_SPECS, LAYER_DIV)):    
        #                     layer  sh      l   u   
        # enumerate data like [ 0, (6, 69), (0, 576) ]
        # enumerate give a number connect layer
        # zip data like [ (6, 69), (0, 576) ]
        # row times column connect valuename sh
        # l means data begin, u means data end
        h5['layer_{}'.format(layer)] = X[:, l:u].reshape((-1, ) + sh)
        # format the layer number in {} 
        # slice the data in X during l~~u
        # reshape convert the 1D data to shape sh---(6, 69)

      #  h5['overflow'] = X[:, -OVERFLOW_BINS:]
      #  h5['energy'] = E.reshape(-1, 1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
    description='Convert cell root files into ML-able HDF5 files')

    parser.add_argument('--in-file', '-i', action="store", required=True,help='input ROOT file')
    parser.add_argument('--out-file', '-o', action="store", required=True,help='output HDF5 file')
    parser.add_argument('--tree', '-t', action="store", required=True,help='input tree for the ROOT file')

    args = parser.parse_args()
    write_out_file(infile=args.in_file, outfile=args.out_file, tree=args.tree)
