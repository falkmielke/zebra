#!/usr/bin/env python3

import sys as SYS           # system control
import os as OS             # operating system control and file operations
import pandas as PD         # data management
import numpy as NP          # numerical analysis
import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API

dpi=300


SYS.path.append('../03_fcas') # makes the folder where the toolbox files are located accessible to python
from Config import config as config # project configuration
from Config import LoadLimbs, ReLoadLimbs # project configuration
SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
import FourierToolbox as FT # Fourier Series toolbox
import PlotToolbox as PT

PT.PreparePlot()


def EqualLimits(ax):
    limits = NP.concatenate([ax.get_xlim(), ax.get_ylim()])
    max_lim = NP.max(NP.abs(limits))

    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])



def MakeSignalFigure():
    fig = PLT.figure(figsize = (1980/dpi, 1080/dpi), dpi=dpi)
    fig.subplots_adjust( \
                              top    = 0.82 \
                            , right  = 0.98 \
                            , bottom = 0.09 \
                            , left   = 0.10 \
                            , wspace = 0.08 # column spacing \
                            , hspace = 0.20 # row spacing \
                            )
    rows = [12]
    cols = [5,2]
    gs = MP.gridspec.GridSpec( \
                              len(rows) \
                            , len(cols) \
                            , height_ratios = rows \
                            , width_ratios = cols \
                            )
    signal_space = fig.add_subplot(gs[0]) # , aspect = 1/4
    signal_space.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    signal_space.set_xlabel(r'stride cycle')
    signal_space.set_ylabel(r'angle')
    signal_space.set_title('time domain', fontsize = 8)
    signal_space.set_xlim([0.,1.])


    frequency_space = fig.add_subplot(gs[1], aspect = 'equal')
    frequency_space.axhline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)
    frequency_space.axvline(0, ls = '-', color = '0.5', lw = 1, zorder = 0)

    frequency_space.set_xlabel(r'$\Re(c_n)$')
    frequency_space.set_ylabel(r'$\Im(c_n)$')
    frequency_space.set_title('frequency domain', fontsize = 8)

    frequency_space.yaxis.tick_right()
    frequency_space.yaxis.set_label_position("right")

    return fig, signal_space, frequency_space



if __name__ == "__main__":
    cyclized_angles = PD.read_csv('../data/cyclized_angles.csv', sep = ';')
    cycles = NP.unique(cyclized_angles['cycle_nr'].values)

    fig, sig_domain, frq_domain = MakeSignalFigure()

    reference = 'wrist'
    joint = 'elbow'

    avg_fsd = []
    for idx in cycles:
        angle = cyclized_angles.loc[cyclized_angles['cycle_nr'] == idx, joint].values
        time = NP.linspace(0., 1., len(angle), endpoint = True)
        sig_domain.plot(time, angle, color = 'darkred', lw = 0.5, alpha = 0.1, zorder = 10)
    #
        fsd = FT.FourierSignal.FromSignal(time, angle, order = 8, period = 1., label = idx)
        frq_domain.plot(fsd._c.iloc[1:, 0], fsd._c.iloc[1:, 1], ls = '-', color = 'darkred', lw = 0.5, alpha = 0.1, zorder = 10)
        avg_fsd.append(fsd._c.values)

    avg_fsd = PD.DataFrame(NP.mean(NP.stack(avg_fsd, axis = 2), axis = 2), columns = FT.re_im, index = NP.arange(9))

    average = FT.FourierSignal.FromDataFrame(avg_fsd, label = 'avg')
    sig_domain.plot(time, average.Reconstruct(x_reco = time, period = 1.), color = 'k', alpha = 0.9, zorder = 50)
    frq_domain.scatter(average._c.iloc[1:, 0], average._c.iloc[1:, 1], s = 3, marker = 'o', color = 'k', alpha = 0.9, zorder = 50)
    frq_domain.plot(average._c.iloc[1:, 0], average._c.iloc[1:, 1], ls = '-', color = 'k', alpha = 0.9, zorder = 50)
    # limbs = LoadLimbs(cycles)
    #

    EqualLimits(frq_domain)
    fig.suptitle(f'zebra {joint} angle - raw')
    fig.tight_layout()
    fig.savefig(f'../figures/zebra_{joint}_raw.svg', transparent = False, dpi = dpi)
    fig.savefig(f'../figures/zebra_{joint}_raw.png', transparent = False, dpi = dpi)
    PLT.show()

    avg_raw = average



    limbs = ReLoadLimbs(cycles)
    references = {idx: lmb[joint] for idx, lmb in limbs.items()}
    joints = {idx: lmb[joint] for idx, lmb in limbs.items()}

    fig, sig_domain, frq_domain = MakeSignalFigure()
    # print (joints)

    time = NP.linspace(0., 1., 101, endpoint = True)

    for wrist in joints.values():
        sig_domain.plot(time, wrist.Reconstruct(x_reco = time, period = 1.), color = 'darkblue', lw = 0.5, alpha = 0.1, zorder = 10)
        frq_domain.plot(wrist._c.iloc[1:, 0], wrist._c.iloc[1:, 1], ls = '-', color = 'darkblue', lw = 0.5, alpha = 0.1, zorder = 10)

    average = FT.ProcrustesAverage( \
                            [jnt for jnt in joints.values()] \
                            , n_iterations = 5, skip_scale = False, post_align = True \
                            )
    # print (average)
    sig_domain.plot(time, average.Reconstruct(x_reco = time, period = 1.), color = 'k', alpha = 0.9, zorder = 50)
    frq_domain.scatter(average._c.iloc[1:, 0], average._c.iloc[1:, 1], s = 3, marker = 'o', color = 'k', alpha = 0.9, zorder = 50)
    frq_domain.plot(average._c.iloc[1:, 0], average._c.iloc[1:, 1], ls = '-', color = 'k', alpha = 0.9, zorder = 50)

    sig_domain.plot(time, avg_raw.Reconstruct(x_reco = time, period = 1.), ls = '--', lw = 0.5, color = 'k', alpha = 0.9, zorder = 50)
    frq_domain.plot(avg_raw._c.iloc[1:, 0], avg_raw._c.iloc[1:, 1], ls = '--', lw = 0.5, color = 'k', alpha = 0.9, zorder = 50)

    EqualLimits(frq_domain)
    fig.suptitle('Zebra Wrist Angle - Aligned')
    fig.tight_layout()
    fig.savefig('../figures/zebra_elbow_wristaligned.svg', transparent = False, dpi = dpi)
    fig.savefig('../figures/zebra_elbow_wristaligned.png', transparent = False, dpi = dpi)
    PLT.show()
