#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20230816

"""
This functions below will take data from aligned (or unaligned) joint angle
systems and perform multivariate analysis.
"""


################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import re as RE             # regular expressions, to extract patterns from text strings
import numpy as NP          # numerical analysis
import pandas as PD         # data management

# load self-made toolboxes
from Config import config as config # project configuration
from Config import ReLoadLimbs, CombineNestedList # project configuration
SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
import EigenToolbox as ET   # Eigenanalysis/PCA
import FourierToolbox as FT # Fourier Series tools


# plotting:
import matplotlib as MP # plotting
import matplotlib.pyplot as PLT # plotting

#_______________________________________________________________________________
# helper variables and functions
re_im = ['re', 'im']



################################################################################
###  Data Preparation                                                        ###
################################################################################

def LimbToDataRow(lmb, all_joints: list, all_components: list) -> NP.array:
    # get a single data row for one limb

    joint_data = []
    for jnt in all_joints:
        # re-arrange the FSD
        joint_fsd = lmb[jnt].Copy()

        # isolate affine components
        # mean, amp, phase = FT.Standardize(joint_fsd)
        mean = FT.Center(joint_fsd)
        amp = FT.Normalize(joint_fsd)
        phase = 0.

        # get (non-affine) components except zero'th (which is zero)
        components = [comp for comp in all_components \
                      if comp > 0]
        nonaffine = joint_fsd.GetComponents(components)

        # assemble data
        params = ['μ|re', 'α|re', 'φ|re'] + [f'{comp}|{ri}' for comp in components for ri in re_im ]
        params = [f'{jnt}|{param}' for param in params]
        values = [mean, amp, phase] + list(nonaffine)
        data_row = PD.Series(values, index = params)

        # append
        joint_data.append(data_row)

    return PD.concat(joint_data)


def LimbsToDataMatrix(limbs: dict) -> PD.DataFrame:
    # convert a dict of limbs to a data matrix

    # get all joints in the data set
    # all_joints = CombineNestedList([ \
    #                                      [key for key in lmb.keys() \
    #                                       if not (key == lmb.reference_joint) \
    #                                       ] \
    #                                    for lmb in limbs.values()])
    all_joints = set.intersection(*[ \
                 {key for key in lmb.keys() \
                  if not (key == lmb.reference_joint) \
                  } \
                   for lmb in limbs.values() \
                  ])

    all_components = CombineNestedList([list(lmb[jnt]._c.index.values) for jnt in all_joints for lmb in limbs.values()])

    # print (all_joints)

    # loop limbs
    data = {}
    for key, lmb in limbs.items():
        # get a single data row
        data_row = LimbToDataRow(lmb, all_joints, all_components)

        # append data
        data[key] = data_row

    # merge the data series
    data = PD.DataFrame(data).T

    # remove 0|im columns # obsolete since affine extraction
    # data.drop(columns = [col for col in data.columns if '|0|im' in col], inplace = True)

    data.index.name = 'cycle_idx'

    # return
    return data


################################################################################
###  Multivariate Analysis                                                   ###
################################################################################
#_______________________________________________________________________________
### Calculation
def PCAnalysis(data: PD.DataFrame, dim: int = None) -> ET.PrincipalComponentAnalysis:

    # data.loc[:, :] = NP.array(data.values, dtype = NP.float64)
    data = data.astype(NP.float64)

    pca = ET.PrincipalComponentAnalysis(data, features = data.columns, standardize = False) # reduce_dim_to = 20

    # print (data.values[:5, :8])
    # rtf = pca.ReTraFo(pca.transformed.values)
    # print (rtf[:5, :8])
    assert NP.allclose(data.values, pca.ReTraFo(pca.transformed.values))


    if dim is not None:
        if dim == 'auto':
            weights = pca.weights
        else:
            weights = pca.weights[:dim]

        pca.ReduceDimensionality(redu2dim = dim, threshold = 0.99)

    print (NP.sum(pca.weights), len(pca.weights))
    print (NP.cumsum(pca.weights))

    # quick check
    rtf = pca.ReTraFo(pca.transformed.values)
    print (NP.subtract(data.values, rtf)[:5, :8])

    return pca


#_______________________________________________________________________________
### Plotting
inch = 2.54
cm = 1/inch
def PlotPCA2D(pca, dims = [0,1], labels = None, figax = None, figargs = {'dpi': 300, 'figsize': (24*cm, 16*cm) }, scatterargs = {} ):
    if figax is not None:
        fig, ax = figax
    else:
        fig = PLT.figure(**figargs)
        ax = fig.add_subplot(1, 1, 1, aspect = 'equal')

    fig.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.99, top = 0.99)

    values = pca.transformed.iloc[:, dims]
    ax.scatter(values.iloc[:, 0], values.iloc[:, 1], **scatterargs)

    if labels is not None:
        for idx, val in values.iterrows():
            ax.text(  val[0], val[1] \
                    , labels.get(idx, idx) \
                    , ha = 'left', va = 'bottom' \
                    , fontsize = 4 \
                    , alpha = 0.8 \
                    )


    ax.axhline(0, color = 'k', ls = '-', lw = 0.5)
    ax.axvline(0, color = 'k', ls = '-', lw = 0.5)

    ax.set_xlabel(f'PC{dims[0]+1} ({100*pca.weights[dims[0]]:0.1f} %)')
    ax.set_ylabel(f'PC{dims[1]+1} ({100*pca.weights[dims[1]]:0.1f} %)')

    return [fig, ax]


def PlotPCA(pca, dims = range(3), labels = None ):

    fig = PLT.figure(**{'dpi': 300, 'figsize': (20*cm, 26*cm) })

    gs = MP.gridspec.GridSpec( \
                                  len(dims) \
                                , 1 \
                                , height_ratios = [1]*len(dims) \
                                , width_ratios = [1] \
                                )


    values = pca.transformed.iloc[:, dims]

    ref_ax = None
    for d in dims:
        if ref_ax is None:
            ax = fig.add_subplot(gs[d])
            ref_ax = ax
        else:
            ax = fig.add_subplot(gs[d], sharex = ref_ax)


        for idx, v in values.iloc[:, d].items():
            ax.axvline(v, color = 'k', ls = '-', lw = 1)

            if labels is not None:
                ax.text(  v, 0.05 \
                        , labels[idx] \
                        , ha = 'left', va = 'bottom' \
                        , fontsize = 8 \
                        , alpha = 0.6 \
                        , rotation = 60 \
                        )

        ax.set_ylim([0., 1.])
        ax.set_yticks([])
        ax.set_ylabel(f'PC{d+1} ({100*pca.weights[d]:0.1f}%)', fontsize = 8)

    return fig


################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    cyclized_angles = PD.read_csv('../data/cyclized_angles.csv', sep = ';')
    cycles = NP.unique(cyclized_angles['cycle_nr'].values)

    # load joint fsd, combined as "limbs"
    limbs = ReLoadLimbs(cycles)
    # print(limbs)

    # convert limb FSD to table format
    # (therein extracting affine components)
    data = LimbsToDataMatrix(limbs)

    # coordination pca, i.e. pca of only non-affine components
    # ['μ', 'α', 'φ']
    affine_cols = [(col.split('|')[1] in ['μ', 'α', 'φ']) \
                   for col in data.columns \
                    ]

    ### (i) posture = affine components
    posture_data = data.loc[:, affine_cols]

    # rename
    ChangeCol = lambda col: f"{col[1]}_{col[0]}"
    posture_data.columns = [ChangeCol(col.split('|')) \
                         for col in posture_data.columns \
                         ]

    # save
    posture_storage = '../data/fcas_posture.csv'
    posture_data.to_csv(posture_storage, sep = ';')
    print (f'affine components stored to {posture_storage}!', " "*16)

    ### (ii) coordination = non-affine components
    # use non-affine components for coordination pca
    coordination_data = data.loc[:, NP.logical_not(affine_cols)]

    # filter joints for PCA
    pca_joints = config['pca_joints']
    coordination_data = coordination_data.loc[:, [ \
                         col for col in coordination_data.columns \
                         if (col.split('|')[0] in pca_joints) \
                        ]]

    # store PCA input
    coordination_data.to_csv('../data/fcas_coordination_raw.csv', sep = ';')

    # Principal Components
    pca = PCAnalysis(coordination_data, dim = 'auto')

    # plot 1: 2D plot
    figax = PlotPCA2D(pca, dims = [0,1] \
                      , scatterargs = dict(s = 20, marker = 'o', color = 'k', alpha = 0.8) \
                      , labels = {} \
                      )
    PLT.show()

    # plot 2: axis-wise
    PlotPCA(pca, dims = range(5))
    PLT.close()

    # store the PCA
    pca_storage = '../data/fcas_coordination_pca.pca'
    pca.Save(pca_storage)
    print (f'coordination PCA stored to {pca_storage}!', " "*16)

    # print (pca.transformed)
    trafo_storage = '../data/fcas_coordination_trafo.csv'
    pca.transformed.to_csv(trafo_storage, sep = ';')
    print (f'transformed data stored to {trafo_storage}!', " "*16)
