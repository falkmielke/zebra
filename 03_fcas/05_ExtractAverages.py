#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20211108

"""
All joint angle Fourier data are joined to a "limb"
and aligned with respect to a reference profile.
"""

################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import pandas as PD         # data management
import numpy as NP          # numerical analysis
import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API



SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
from Config import config as config # project configuration
from Config import LoadLimbs, StoreLimbs # project configuration
import FourierToolbox as FT # Fourier Series toolbox



################################################################################
### Procedure                                                                ###
################################################################################

def GetAverages(some_limbs \
              , n_iterations = 5 \
              ):
    # GPA-like alignment of joint angle profiles

    # first, get the average of the reference joint...

    for jnt in config['joint_selection']:
        print (f'averaging {jnt} angle profile...', ' '*16, end = '\r', flush = True)
        average = FT.ProcrustesAverage( \
                            [lmb[jnt] for lmb in some_limbs.values() \
                             if jnt in lmb.keys()] \
                            , n_iterations = n_iterations, skip_scale = False, post_align = False \
                            )._c

        print (average)
        average.to_csv(f'../data/averages/{jnt}.csv', sep = ';')

        print (f'{jnt} averaging done.', ' '*32)

################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    # load joint fsd, combined as "limbs"

    cyclized_angles = PD.read_csv('../data/cyclized_angles.csv', sep = ';')
    cycles = NP.unique(cyclized_angles['cycle_nr'].values)

    limbs = LoadLimbs(cycles)
    # print(limbs)
    # time = limbs['c00007']['knee'].time
    # PLT.plot(time, limbs['c00007']['knee'].Reconstruct(time))

    # alignment, relative to a reference joint
    reference_joint = config['reference_joint']
    GetAverages(limbs \
              , n_iterations = 5 \
              )
