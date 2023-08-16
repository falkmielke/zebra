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

def Alignment(  some_limbs \
              , reference_joint \
              , n_iterations = 5 \
              , superimposition_kwargs = dict(skip_shift = False \
                                              , skip_scale = False \
                                              , skip_rotation = False \
                                              ) \
              , plot = True \
              ):
    # GPA-like alignment of joint angle profiles

    if plot:
        # pre-/post alignment plot
        fig = PLT.figure()
        ax = fig.add_subplot(2,1,1)
        for idx, lmb in limbs.items():
            lmb.Plot(ax, subset_joints = [reference_joint, test_joint])

    # first, get the average of the reference joint...
    print (f'averaging {reference_joint} angle profile...', ' '*16, end = '\r', flush = True)
    average = FT.ProcrustesAverage( \
                            [lmb[reference_joint] for lmb in some_limbs.values() \
                             if reference_joint in lmb.keys()] \
                            , n_iterations = n_iterations, skip_scale = True, post_align = False \
                            )
    print ('reference averaging done.', ' '*32)

    # ... then align all the limbs to it
    skipped = []
    for label, lmb in some_limbs.items():
        # # standardize the shift and amplitude of the focal joint to give relative values
        # lmb.PrepareRelativeAlignment(focal_joint, skip_rotate = True, skip_center = skip_precenter, skip_normalize = skip_prenorm)
        print (f'aligning {label} ...', ' '*16, end = '\r', flush = True)

        if reference_joint not in lmb.keys():
            skipped.append(label)
            continue

        # align all joints, based on the reference
        lmb.AlignToReference(  reference_joint \
                             , average \
                             , superimposition_kwargs = superimposition_kwargs \
                             )

    print ('skipped', skipped, ' '*32)

    if plot:
        print (f'plotting...', ' '*16, end = '\r', flush = True)
        # show result after alignment
        ax = fig.add_subplot(2,1,2)
        for idx, lmb in limbs.items():
            lmb.Plot(ax, subset_joints = [reference_joint, test_joint])
        PLT.show()

    print ('FCAS done!', ' '*16)

    return average


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
    superimposition_kwargs = config['superimposition_choice']
    Alignment(  limbs \
              , reference_joint = reference_joint \
              , n_iterations = 5 \
              , superimposition_kwargs = superimposition_kwargs \
              , plot = False \
              )

    # store results
    StoreLimbs(limbs)
    print ('superimposed limbs stored.', ' '*24)
