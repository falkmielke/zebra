#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20230816


"""
Here, we step into the frequency domain, transforming the
joint angle profiles with Fourier Series.
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
import FourierToolbox as FT # Fourier Series toolbox


#_______________________________________________________________________________
# helper variables
xy = ['x', 'y'] # 3d coordinate shorthand
reim = ['re', 'im'] # real/imaginary coordinate shorthand

################################################################################
### Structural Settings                                                      ###
################################################################################
def CalculateFourierCoefficients(joint_angles, order = 3, plotfile = None):
    # converts the joint angles into the frequency domain

    # required below: all relevant fourier coefficients for each joint angle
    indices = [(coeff, ri) for coeff in range(order+1) for ri in reim if not (f'{coeff}_{ri}' == '0_im')]

    # prepare a storage data frame
    fourier_coefficients = PD.DataFrame(index = PD.MultiIndex.from_tuples(indices))
    fourier_coefficients.index.names = ['coeff', 're_im']


    # common time
    if False:
        time = NP.linspace(0., 1., len( joint_angles.index.values ))
        period = 1.
    else:
        # TEST: try actual time (shouldn't make difference, but more intuitive)
        time = joint_angles.index.values.astype(float)
        time -= time[0]
        time /= NP.max(time)
        period = time[-1] - time[0]

    # loop all joint angles
    for jnt, angle in joint_angles.T.iterrows():

        # nonnan
        nonnans = NP.logical_not(NP.isnan(angle.values))
        if not NP.any(nonnans):
            continue

        # fourier series decomposition
        fsd = FT.FourierSignal.FromSignal(time = time[nonnans] \
                                          , signal = angle.values[nonnans] \
                                          , order = order \
                                          , period = period \
                                          , label = jnt \
                                          )

        # get coefficients
        coefficients = fsd.GetCoeffDataFrame()

        # if jnt == 'mcarp':
        #     print(coefficients)

        # append storage data frame
        fourier_coefficients[jnt] = NP.array([coefficients.loc[coeff, ri] for coeff, ri in indices])

        if plotfile is not None:
            PLT.plot(time, angle.values, color = 'k', lw = 1, ls = '--', alpha = 0.6, label = None)
            PLT.plot(time, fsd.Reconstruct(time), lw = 1., alpha = 1., label = jnt)

    if plotfile is not None:
        PLT.legend(ncol = 2)
        PLT.gca().set_xlim([0.,1.])
        PLT.savefig(plotfile, dpi = 300, transparent = False)


        PLT.close()

    return fourier_coefficients




def TransformCycles():
    # the procedure of processing joint angle profiles

    cyclized_angles = PD.read_csv('../data/cyclized_angles.csv', sep = ';')
    cycles = NP.unique(cyclized_angles['cycle_nr'].values)

    # loop cycles
    count = 0
    for cycle_idx in cycles:

        # data labeling
        label = f'c{cycle_idx:02.0f}'
        print (f'({count}/{len(cycles)})', label, ' '*16, end = '\r', flush = True)
        count += 1

        # load joint angle profiles
        joint_angles = cyclized_angles.loc[cyclized_angles['cycle_nr'].values == cycle_idx, :].copy()
        cyclized_angles['time'] = NP.arange(cyclized_angles.shape[0])/config['fps']

        # pdf file to store inspection plot
        plot_file = f"inspection_plots/c{cycle_idx:02.0f}.png"

        ## Fourier Series
        selected_joints = config['joint_selection']
        order = config['fourier_order']
        fourier_coefficients = CalculateFourierCoefficients( \
                                          joint_angles.loc[:, selected_joints] \
                                        , order = order \
                                        , plotfile = plot_file \
                                        )

        # store the results
        fsd_file = f"../data/fsd/c{cycle_idx:02.0f}.csv"
        fourier_coefficients.to_csv(fsd_file, sep = ';')

    print (f"Fourier conversion done ({len(cycles)} cycles)", " "*16)


################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    # trafo procedure
    TransformCycles()
