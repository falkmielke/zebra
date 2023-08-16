#!/usr/bin/env python3


################################################################################
### Joint Angle Extraction                                                   ###
################################################################################
__author__      = "Falk Mielke"
__date__        = 20230816

"""
This script and collection of functions is used to extract joint angle profiles
from previously prepared stride cycle kinematics.

Sub-Routines:
- load stride cycle data
- calculate joint angle profiles
- remove end-start difference ("make cyclical")


The process starts at the main script below (if __name__ == "__main__").
Questions are welcome (falkmielke.biology@mailbox.org)

"""

################################################################################
### Libraries                                                                ###
################################################################################
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import re as RE             # regular expressions, to extract patterns from text strings
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import scipy.signal as SIG  # signal processing (e.g. smoothing)
import scipy.interpolate as INTP # interpolation

import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API

# load self-made toolboxes
SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
from Config import config as config # project configuration
import QuaternionGeometryToolbox as QGT # point superimposition tools (e.g. Procrustes)
import FourierToolbox as FT


#_______________________________________________________________________________
# helper variables
xyz = ['x', 'y', 'z'] # 3d coordinate shorthand
xy = ['x', 'y'] # 2d coordinate shorthand


## joint angle profile rectification
def CyclicWrapInterpolateAngle(angle: NP.array, skip_wrap: bool = False) -> NP.array:
    # three steps in one:
    #  - repeat cyclic trace
    #  - wrap angular data
    #  - interpolate NAN gaps

    # replication of the trace
    time = NP.linspace(0., 3., 3*len(angle), endpoint = False)

    signal = NP.concatenate([angle]*3)

    # exclude nans
    nonnans = NP.logical_not(NP.isnan(signal))

    # wrapping: some signals can jump the 2π border
    if NP.any(NP.abs(NP.diff(signal[nonnans])) > NP.pi):
        wrapped_signal = signal.copy()
        wrapped_signal[wrapped_signal < 0] += 2*NP.pi
    else:
        wrapped_signal = signal

    # rbf interpolation of NANs
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
    intp_signal = INTP.Rbf(time[nonnans] \
                             , wrapped_signal[nonnans] \
                             , function = 'thin_plate' \
                             )(time)

    # cut middle part
    angle_intp = intp_signal[len(angle):2*len(angle)]

    return angle_intp


################################################################################
### Kinematics To Joint Angles                                               ###
################################################################################
def ConvertCycleData() -> PD.DataFrame:
    # load stride cycle data and calculate joint angle profiles
    # Note: this function must treat time points/frames independently (because multiple cycles are used)

    ### load kinematic data
    kinematics_raw = PD.read_csv('../data/all_cycles.csv', sep = ';')

    # joint definitions
    joint_definitions = config['joints']

    ### Joint Angle Extraction
    # perform a deep copy of the input data
    # this allows us to modify the structure of the data without changing the original
    kinematics = kinematics_raw.copy()

    # add fake landmarks to get segment angles
    for lm in [0, 99]:
        kinematics.loc[:, f'pt{lm:02.0f}_x'] = 0.
        kinematics.loc[:, f'pt{lm:02.0f}_y'] = 0. if lm == 0 else 1.


    # prepare output data frame
    joint_angles = PD.DataFrame(index = kinematics.index.copy())
    copy_columns = ['cycle_nr', 'folder', 'frame_nr']
    joint_angles.loc[:, copy_columns] = kinematics_raw.loc[:, copy_columns].copy()

    # loop through all joints and calculate angles
    for joint_nr, (landmark_numbers, joint) in joint_definitions.items():
        print (f'extracting {joint} angle ({joint_nr}/{len(joint_definitions)}) ', ' '*16, end = '\r', flush = True)

        landmarks = [config['landmarks'][lmnr] for lmnr in landmark_numbers]

        # get positions of each landmark
        positions = [kinematics.loc[:, [f'{lm}_{coord}' for coord in xy]].values for lm in landmarks]


        # calculate vectors = differences between two points, pointing distally
        proximal_vector = positions[1] - positions[0]
        distal_vector = positions[3] - positions[2]

        # get joint angle:
        #  - zero at straight joint | +/-π at fully folded configuration
        #  - EXCEPT head and shoulder: zero is hanging down
        #  - positive at CCW rotation, negative at CW rotation
        #  - remember: all movements rightwards
        joint_ang = QGT.WrapAnglePiPi( \
                     [QGT.AngleBetweenCCW(pvec, dvec) \
                      for pvec, dvec in zip(proximal_vector, distal_vector)] \
                    )

        # ARCHIVE: # calculate joint angle 3D (quaternions)
        # joint_ang = [QGT.FindQuaternionRotation(q_prox, q_dist).angle() \
        #             for q_prox, q_dist \
        #             in zip(QGT.QuaternionFrom3D(proximal_vector), QGT.QuaternionFrom3D(distal_vector)) \
        #             ]

        # store the joint angle
        joint_angles[joint] = joint_ang

    print (f'joint angle extraction done ({len(joint_definitions)} joints) ', ' '*16)

    # return the raw joint angles
    return joint_angles



################################################################################
### Joint Angle Cyclization                                                  ###
################################################################################
### smoothing angle progiles
def SmoothTrace(time: NP.array, angle: NP.array, *args, **kwargs) -> NP.array:
    # attempting Savitzky-Golay filtering
    # i.e. piecewise polynomial smoothing
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    # relevant input parameters:
    # window_length: how wide the smoothed piece
    # polyorder: polynomial order to smooth
    # mode: edge handling, e.g. "wrap"
    # deriv/delta: smooth by derivatives

    # intp_angle = IOT.CyclicWrapInterpolateAngle(angle)
    # THIS IS NOW DONE PER DEFAULT

    # # replication of the trace
    # time = NP.linspace(0., 3., 3*len(angle), endpoint = False)
    # # time = NP.concatenate([time.ravel()]*3)

    # signal = NP.concatenate([angle]*3)

    # # exclude nans
    # nonnans = NP.logical_not(NP.isnan(signal))

    # # wrapping: some signals can jump the 2π border
    # if NP.any(NP.abs(NP.diff(signal[nonnans])) > NP.pi):
    #     wrapped_signal = signal.copy()
    #     wrapped_signal[wrapped_signal < 0] += 2*NP.pi
    # else:
    #     wrapped_signal = signal

    # # rbf interpolation of NANs
    # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
    # intp_signal = INTP.Rbf(time[nonnans] \
    #                          , wrapped_signal[nonnans] \
    #                          , function = 'thin_plate' \
    #                          )(time)


    angle_smoothed  = SIG.savgol_filter(angle, *args, **kwargs)

    if False:
        PLT.plot(time, signal, lw = 2, alpha = 0.8)
        PLT.plot(time, smooth_signal, lw = 0.5, ls = '-')
        PLT.gca().axvline(time[len(angle)])
        PLT.gca().axvline(time[2*len(angle)])
        PLT.show();


    return angle_smoothed


#___________________________________________________________________________________________________
### start/end correction of a singnal
def MakeCyclical(angle: NP.array) -> NP.array:
    # because there can still be a small difference from end to start, angle traces are forced to be cyclical.

    # # first, a bit of gentle smoothing
    # # smoothed_trace = SmoothTrace(angle, order = 12)
    # smoothed_trace = SmoothTrace(  time \
    #                              , angle \
    #                              , window_length = 9 \
    #                              , polyorder = 5 \
    #                              , mode = 'wrap' \
    #                              )

    # # take a few frames before and after the end
    # frame = 8

    # # get end-start difference
    # endstart_difference = NP.nanmean(smoothed_trace[-frame:]) - NP.nanmean(smoothed_trace[:frame])
    endstart_difference = angle[-1] - angle[0]

    #... and spread correction of the difference linearly over the whole stride cycle
    return NP.subtract(angle, NP.linspace(0, endstart_difference, len(angle), endpoint = True))


#_______________________________________________________________________________
def CyclizeSingleStride(cycle_nr: int, stride_angles_raw: PD.DataFrame) -> PD.Series:
    # make joint profiles cyclical (NOT in place)
    # by spreading the end-start difference over the stride cycle
    # returns cyclized angles and difference per joint

    stride_angles = stride_angles_raw.copy()

    # store the "time" dimension (required for smoothing)
    time = NP.arange(stride_angles.shape[0]) / config['fps']


    difference = {}
    # loop angles
    for col in stride_angles.columns:

        # take one raw angular profile
        angle = stride_angles.loc[:, col].values.ravel().copy()

        # cyclic interpolation
        angle = CyclicWrapInterpolateAngle(angle.ravel())

        # end-start correction
        angle_cyclic = MakeCyclical(angle.copy())

        PLT.plot(angle - angle[0], label = col)
        # PLT.plot(angle_cyclic)

        # remove jumps
        # print (NP.max(NP.abs(NP.diff(angle_cyclic))))
        nonnans = NP.logical_not(NP.isnan(angle_cyclic))
        if NP.max(NP.abs(NP.diff(angle_cyclic[nonnans]))) > NP.pi:
            angle_cyclic[angle_cyclic < 0.] += 2*NP.pi

        # correct the column
        stride_angles.loc[:, col] = angle_cyclic

        difference[col] = angle_cyclic[-1] - angle[-1]

    PLT.legend(loc = 'best')
    PLT.axhline(0, ls = ':', color = 'k', alpha = 0.6)
    ax = PLT.gca()
    ax.set_xlim([0, stride_angles.shape[0]])
    ax.set_xlabel('frame')
    ax.set_ylabel('angle (rad)')
    PLT.savefig(f'cycle_differences/c{cycle_nr:02.0f}.png', dpi = 300, transparent = False)
    PLT.close()
    return stride_angles, difference


def CyclizeStrides(joint_angles: PD.DataFrame) -> dict:
    # take a joined joint data frame with joint angles over time and stride cycles
    # loop cycles and make them cyclical.
    # crazy shit.


    # loop all strides
    cycle_list = NP.unique(joint_angles['cycle_nr'].values)
    cycle_differences = {}
    cyclized_angles = []
    for nr, cycle_idx in enumerate(cycle_list):
        # print the label
        print (f'({nr}/{len(cycle_list)}), cycle {cycle_idx}', ' '*16, end = '\r', flush = True)

        # take angles for each stride
        stride_angles = joint_angles.loc[joint_angles['cycle_nr'].values == cycle_idx\
                                         , : \
                                         ].copy()

        # apply cyclization procedure (changing `stride_angles` in place)
        cyclized, diffs = CyclizeSingleStride(cycle_idx, stride_angles.loc[:, [ang for _, ang in config['joints'].values()]])
        if (NP.sum([NP.abs(val) for val in diffs.values()]) > 0.4):
            print ('skipping cycle:', cycle_idx)
            continue
        # print (cycle_idx, diffs)
        cycle_differences[cycle_idx] = diffs

        # # store the angles for this stride
        # store_file = f"../data/cycle_angles/c{cycle_idx:02.0f}.csv"
        # cyclized.to_csv(store_file, sep = ';')

        stride_angles.loc[:, cyclized.columns] = cyclized.values
        cyclized_angles.append(stride_angles.iloc[:-1, :])

    cyclized_angles = PD.concat(cyclized_angles)
    cyclized_angles.to_csv('../data/cyclized_angles.csv', sep = ';')

    print (f"cycle cyclization done ({len(cycle_list)} cycles)", " "*16)

    cycle_diffs = PD.DataFrame.from_dict(cycle_differences).T
    cycle_diffs['total'] = NP.sum(NP.abs(cycle_diffs.values), axis = 1)
    # print (cycle_diffs)
    PLT.hist(cycle_diffs['total'], bins = 16)
    PLT.show()

    # store output
    cycle_diffs.to_csv(f'../data/cyclediffs.csv', sep = ';')

    print ('cyclic cycles stored!')




################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    # extract joint angles
    joint_angles = ConvertCycleData()
    joint_angles.to_csv('../data/joint_angles.csv', sep = ';')
    # print (joint_angles)

    # for angle in joint_angles.columns:
    #     PLT.plot(joint_angles[angle].values, label = angle)
    #     PLT.legend(loc = 'best')

    # PLT.show()

    # make cyclical
    cycle_diffs = CyclizeStrides(joint_angles)
