#!/usr/bin/env python3

import sys as SYS
import numpy as NP
import pandas as PD
import matplotlib as MP
import matplotlib.pyplot as PLT

SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
import PlotToolbox as PT

def LoadAllData():
     spatiotemporals = PD.read_csv('../data/master_data.csv', sep = ';')
     spatiotemporals.index = [f'c{cy:02.0f}' for cy in spatiotemporals['cycle_nr'].values]
     spatiotemporals.index.name = 'cycle_index'
     coordination = PD.read_csv('../data/fcas_coordination_raw.csv', sep = ';').set_index('cycle_idx', inplace = False)
     posture = PD.read_csv('../data/fcas_posture.csv', sep = ';').set_index('cycle_idx', inplace = False)

     all_data = spatiotemporals.join(posture, how = 'left').join(coordination, how = 'left')

     return all_data

def DurationHistogram(data):
    fig, gs = PT.MakeFigure(dimensions = [16.5, 10], dpi = 300)
    ax = fig.add_subplot(gs[0])

    bins = NP.arange(30, 39)
    ax.hist(data['n_frames'].values, bins = bins, align = 'mid', facecolor = '0.7', edgecolor = 'k')
    ax.set_xticks(bins[:-1]+0.5)
    ax.set_xticklabels(bins[:-1])
    ax.set_title('Zebra Data: Stride Cycle Duration')
    ax.set_ylabel('count')
    ax.set_xlabel('stride cycle duration (frames)')
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig('../figures/zebra_duration.pdf', dpi = 300, transparent = False)
    PLT.show()


if __name__ == "__main__":
    all_data = LoadAllData()
    print (all_data.columns)

    DurationHistogram(all_data)
