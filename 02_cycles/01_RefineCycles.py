#!/usr/bin/env python3

import os as OS
import sys as SYS
import time as TI
import numpy as NP
import pandas as PD
import scipy.signal as SIG
import matplotlib as MP
import matplotlib.pyplot as PLT
import skimage as SKI
from skimage import feature as FEAT
from tqdm import tqdm

SYS.path.append('../toolboxes')
import SuperimpositionToolbox as ST

xy = ['x', 'y']

pd_threshold = 0.0005


def FindLocalCenterMinimum(image, pos = None):
    if pos is None:
        pos = NP.array(image.shape) // 2
        print ('start:', pos)

    wl = 3
    window = image[pos[0]-wl:pos[0]+wl+1, pos[1]-wl:pos[1]+wl+1]

    if NP.all(NP.isnan(window)):
        print ('nan escape')
        return pos
    print (window, NP.nanargmin(window))
    d_pos = NP.unravel_index(NP.nanargmin(window), window.shape)
    print (d_pos)
    if NP.sum(NP.abs(d_pos - wl*NP.ones((2,)))) == 0:
        # found minimum
        print ('found minimum')
        return pos
    pos = pos + d_pos - wl
    print ('new:', d_pos, pos)
    if (pos[0] == 0) or (pos[0] + 1 == image.shape[0]) \
       or (pos[1] == 0) or (pos[1] + 1 == image.shape[1]):
        # escape border
        print ('border escape')
        return None

    return FindLocalCenterMinimum(image, pos)


def AutoCorr(signal):
    result = NP.correlate(signal, signal, mode='full')
    return result[result.size//2:]


def RefineCycles(traces):

    exclude_landmarks = [  'snout_x', 'ear_x' \
                         ] # exclude head module
    landmarks = ['_'.join(col.split('_')[:-1]) for col in traces.columns if (col[-2:] == '_x') \
                  and not (col in exclude_landmarks) \
                  ]


    # store output as 2d image
    # range_t1 = NP.arange(t1-interval_t1, t1+interval_t1+1)
    # range_t2 = NP.arange(t2-interval_t2, t2+interval_t2+1)
    range_t1 = NP.arange(0, 8)
    range_t2 = NP.arange(traces.shape[0]-10, traces.shape[0])
    pd_image = NP.empty((len(range_t1), len(range_t2)))
    pd_image[:,:] = NP.nan

    GetConfiguration = lambda t: NP.stack([ \
                                            traces.loc[:, [f'{lm}_{coord}' for lm in landmarks]].iloc[t, :] \
                                            for coord in xy \
                                           ], axis = 1)

    procrustes_distances = {}
    for i_ref, t_ref in enumerate(range_t1):
        # the reference configuration
        ref_configuration = GetConfiguration(t_ref)
        ref_configuration = ST.Standardize(ref_configuration)

        for i_candidate, t_candidate in enumerate(range_t2):
            # the candidate configuration
            candidate_configuration = GetConfiguration(t_candidate)
            candidate_configuration = ST.Standardize(candidate_configuration)

            # procrustes!
            transformed, pd, _ = ST.Procrustes(  ref_configuration \
                                               , candidate_configuration)

            pd_image[i_ref, i_candidate] = pd


    # print (pd_image)

    # find local minima in the image range
    local_minima = FEAT.peak_local_max(-pd_image+NP.nanmax(pd_image) \
                                       , min_distance = 2 \
                                       , exclude_border = False \
                                       )
    center_dist = NP.sum(NP.abs(local_minima - NP.array(pd_image.shape).reshape((1,-1))), axis = 1)
    # print (local_minima, NP.argmin(center_dist))
    best_idx = local_minima[NP.argmin(center_dist), :]
    plotting = False
    if plotting:
        PLT.imshow(pd_image)
        PLT.axhline(best_idx[0])
        PLT.axvline(best_idx[1])
        PLT.show()
        pause
    pd_min = pd_image[best_idx[0],best_idx[1]]
    # print (pd_min)

    trace_refined = traces.iloc[range_t1[0] + best_idx[0]: range_t2[0] + best_idx[1] + 1, :].copy()
    # !!! going full circle, i.e. including first frame of the next cycle

    # plot configurations
    if True:
        cycle_idx = traces.loc[:, 'cycle_nr'].values[0]
        episode = traces.loc[:, 'folder'].values[0]

        start_config = ST.Standardize(GetConfiguration(range_t1[0] + best_idx[0]))
        end_config = ST.Standardize(GetConfiguration(range_t2[0] + best_idx[1]))

        dpi = 300
        fig = PLT.figure(dpi = dpi)
        ax = fig.add_subplot(1,1,1, aspect = 'equal')
        ax.plot(start_config[:,0], -1*start_config[:,1], marker = 'o', color = 'k')
        ax.plot(end_config[:,0], -1*end_config[:,1], marker = 'x', color = 'darkblue' if pd_min < pd_threshold else 'darkred')
        ax.set_title(f'{episode}, cy {cycle_idx}, pd {pd_min:.6f}')
        fig.savefig(f'cycle_procrustes/{episode}_cycle{cycle_idx}.png', dpi = dpi, transparent = False)
        PLT.close()


    return pd_min, trace_refined



if __name__ == "__main__":

    # combine kinematic and raw footfall data
    kine_files = [ \
                     '../data/20230814_frames_ep1.csv' \
                   , '../data/20230815_frames_ep2.csv' \
                   , '../data/20230815_frames_ep3.csv' \
                   ]
    footfall_files = [\
                     '../data/20230815_frames_ep1_footfalls.csv' \
                   , '../data/20230815_frames_ep2_footfalls.csv' \
                   , '../data/20230815_frames_ep3_footfalls.csv' \
                  ]

    cycle_kinematics = []
    cycle_nr = 0
    for kf, fff in zip(kine_files, footfall_files):

        footfalls = PD.read_csv(fff, sep = ';').set_index('frame_nr', inplace = False).loc[:,'snout_x'].isna()
        max_frame = footfalls.shape[0]
        footfalls = NP.where(NP.logical_not(footfalls))[0]
        cycles = NP.stack([footfalls[:-1]-5, 5+footfalls[1:]], axis = 1)
        #
        # min/max
        cycles[cycles<0] = 0
        cycles[cycles>max_frame] = max_frame

        kinematics = PD.read_csv(kf, sep = ';')
        kinematics['folder'] = kinematics['folder'].str.replace('D:/Falk/19_zebra_video/frames_', '')

        for start_end in cycles:
            # print ('_'*40)
            # print (start_end)
            cycle_nr += 1

            this_cycle = kinematics.loc[start_end[0]:start_end[1], :].copy()
            this_cycle.loc[:, 'cycle_nr'] = cycle_nr
            # print (this_cycle)
            cycle_kinematics.append(this_cycle)

    cycle_kinematics = PD.concat(cycle_kinematics)


    cycle_numbers = NP.unique(cycle_kinematics['cycle_nr'].values)
    cycles = []
    for cycle_nr in tqdm(cycle_numbers):

        traces = cycle_kinematics.loc[cycle_kinematics['cycle_nr'].values == cycle_nr, :]
        # print(traces)

        # FindFootfallPatterns(traces, f'{video_file}, {idx}')
        pd, refined_trace = RefineCycles(traces)

        # some cycles are not exactly cyclic
        if pd < pd_threshold:
            cycles.append(refined_trace)

    print (f'{len(cycles)} cycles found!')
    cycles = PD.concat(cycles)
    cycles.to_csv('../data/all_cycles.csv', sep = ';')
