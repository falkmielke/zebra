#!/usr/bin/env python3

__author__      = "Falk Mielke"
__date__        = 20230816

"""
Extract spatiotemporal parameters.
"""
################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import re as RE             # regular expressions (string ragging)
import shelve as SH         # storing dict
# import openpyxl as XL       # handling excel files
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import configparser as CNP  # config to file
import scipy.signal as SIG  # signal processing (e.g. smoothing)
import matplotlib as MP     # plotting, low level API
import matplotlib.pyplot as PLT # plotting, high level API

# load self-made toolboxes
SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
import EigenToolbox as ET    # PCA

#_______________________________________________________________________________
# helper variables
xy = ['x', 'y'] # 2d coordinate shorthand

def CalculateDistanceCovered(kine: PD.DataFrame) -> float:
    # get the average distance covered by the landmarks

    landmarks = kine.columns.remove_unused_levels().levels[0].values
    values = NP.stack([kine.loc[:, lm].values \
                       for lm in landmarks \
                       ], axis = 2)
    values = NP.mean(values, axis = 2)

    distance = NP.subtract(values[-1, :], values[0, :])
    distance = NP.sqrt(NP.sum(NP.power(distance, 2)))

    return distance


def CalculateClearance(kine: PD.DataFrame, TraFo: object = lambda x: x) -> float:
    # compute the clearance from a swing of reference landmarks
    landmarks = ['shoulder', 'hoof']
    values = NP.stack([kine.loc[:, [f'{lm}_{coord}' for coord in xy]].values \
                       for lm in landmarks \
                       ], axis = 2)
    distance = NP.diff(values, axis = 2)
    distance = NP.sqrt(NP.sum(NP.power(distance, 2), axis = 1))
    # flexed, extended = NP.percentile(distance, [5, 95])
    flexed, extended = NP.percentile(distance, [10, 90])

    # return clearance
    return TraFo((extended - flexed) / extended)


################################################################################
### Update Predictors                                                        ###
################################################################################
def AppendAnalysisData() -> PD.DataFrame:
    # take the raw kinematics files and extract:
    # - duty factor
    # - size proxy (shank length)
    # - speed, size-corrected speed
    # - stride duration/frequency
    # - dimensionless spatiotemporal parameters
    #
    # but: most values are already calculated!
    # print (master_data.sample(3).T)
    # missing: spatial calibration (not used); dim.less

    # load kinematics data
    kinematics = PD.read_csv( '../data/all_cycles.csv', sep = ';' )
    # print (kinematics)
    angles = PD.read_csv( '../data/cyclized_angles.csv', sep = ';' )

    # optical_flow = {ep: PD.read_csv('optical_flow_ep{ep}.csv', sep = ';').set_index('frame_nr', inplace = False) for ep in [1,2,3]}

    cycles = NP.unique(kinematics['cycle_nr'].values)

    master_data = PD.DataFrame(index = cycles)
    master_data.index.name = 'cycle_nr'

    ### back length
    for col in ['l_back', 'l_limb', 'n_frames', 'duration', 'stride_distance', 'speed', 'dutyfactor', 'clearance']:
        master_data['l_limb'] = NP.nan

    GetCol = lambda data, col: data.loc[:, [f'{col}_{coord}' for coord in xy]].values
    EuclidColumns = lambda data, col1, col2: NP.sum(NP.abs(NP.subtract(GetCol(data, col1), GetCol(data, col2))), axis = 1)

    for idx in cycles:
        kine = kinematics.loc[kinematics['cycle_nr'].values == idx, :]
        angl = angles.loc[angles['cycle_nr'].values == idx, :]
        episode = int(kine.loc[:, 'folder'].values[0].replace('ep', ''))

        ### spatial references
        l_back = EuclidColumns(kine, 'withers', 'croup')
        # print (idx, NP.mean(l_back), NP.std(l_back))
        master_data.loc[idx, 'l_back'] = NP.mean(l_back)

        limb_landmarks = ['shoulder', 'elbow', 'carpal', 'ankle', 'hoof']
        l_limb = 0
        for p1, p2 in zip(limb_landmarks[:-1], limb_landmarks[1:]):
            l_limb += NP.mean(EuclidColumns(kine, p1, p2))

        master_data.loc[idx, 'l_limb'] = l_limb

        ### stride duration
        fps = 25.0
        n_frames = kine.index.values[-1] - kine.index.values[0]
        master_data.loc[idx, 'n_frames'] = n_frames
        master_data.loc[idx, 'duration'] = n_frames / fps

        ### stride distance
        distance = []
        for lm in ['withers', 'croup']:
            positions = GetCol(kine, lm)
            # print (lm, positions[[0,-1], :])
            dist = positions[-1, 0] - positions[0, 0]
            flow = -16*n_frames # TODO get the real data
            # flow = NP.sum(optical_flow[episode].loc[kine['frame_nr'].values, 'flow'].values)
            distance.append(NP.abs(dist + flow))

        # print (idx, distance)
        master_data.loc[idx, 'stride_distance'] = NP.mean(distance) / master_data.loc[idx, 'l_back']
        master_data.loc[idx, 'speed'] = master_data.loc[idx, 'stride_distance'] / master_data.loc[idx, 'duration']

        ### clearance
        master_data.loc[idx, 'clearance'] = CalculateClearance(kine)

        ### dutyfactor
        master_data.loc[idx, 'dutyfactor'] = NP.sum(NP.diff(angl.loc[:, 'forelimb'].ravel()) > 0.)/n_frames

    return master_data





    for col in ['distance_m', 'clearance_fl', 'clearance_hl']:
        master_data.loc[:, col] = NP.nan
    for idx, row in master_data.iterrows():
        kine = kinematics.loc[idx, :]
        kine.index = (kine.index.values - NP.min(kine.index.values)) \
                   / eval(config['video']['framerate'])

        ### distance_m
        master_data.loc[idx, 'distance_m'] \
            = CalculateDistanceCovered(kine.loc[: \
                                     , [config['landmarks'][str(lmnr)] \
                                        for lmnr in eval(config['analysis']['speedref_landmarks'])] \
                                      ] / row['px_per_m'])

        ### clearance (f/h)
        for limb in ['fl', 'hl']:
            master_data.loc[idx, f'clearance_{limb}'] \
                = CalculateClearance(kine.loc[: \
                                         , [config['landmarks'][str(lmnr)] \
                                            for lmnr in eval(config['analysis'][f'{limb}_clearance_landmarks'])] \
                                        ] / row['px_per_m'] \
                                     # , TraFo = eval(config['analysis']['clearance_trafo']) \
                                     )

    # print (master_data.sample(3).T)

    if False: # plot clearance distribution
        for limb in ['fl', 'hl']:
            PLT.hist(master_data[f'clearance_{limb}'].values, bins = 100, histtype = 'step', label = limb)
        PLT.legend()
        PLT.show()

    gravitational_acceleration = 10. # m/s²
    master_data.loc[:, 'size_ref'] = master_data.loc[:, 'l_animal_m'].values
    master_data.loc[:, 'time_ref'] = NP.sqrt(master_data.loc[:, 'size_ref'].values.astype(float) / gravitational_acceleration)

    # dimensionless measures
    master_data.loc[:, 'distance_diml'] = master_data.loc[:, 'distance_m'].values \
                                       / master_data.loc[:, 'size_ref'].values
    master_data.loc[:, 'duration_diml'] = master_data.loc[:, 'duration_s'].values \
                                       / master_data.loc[:, 'time_ref'].values
    master_data.loc[:, 'frequency_diml'] = 1./master_data.loc[:, 'duration_diml'].values
    master_data.loc[:, 'speed_diml'] = master_data.loc[:, 'speed_m_s'].values \
                                    / NP.sqrt(master_data.loc[:, 'size_ref'].values.astype(float) * gravitational_acceleration)
    # master_data.loc[:, 'speed_diml2'] = master_data.loc[:, 'distance_diml'].values \
    #                                  / master_data.loc[:, 'duration_diml'].values
    # print (NP.diff(master_data.loc[:, ['speed_diml', 'speed_diml2']].values, axis = 1)) # approx zero :)

    # print (master_data.sample(5).T)

    # morphometrics PCA
    pca_features = ['l_back_m', 'l_animal_m', 'l_forelimb_m', 'l_hindlimb_m']
    pca_data = master_data.loc[:, pca_features].dropna()
    pca_data = pca_data.astype(float)

    morpho_pca = ET.PrincipalComponentAnalysis(pca_data, pca_features, standardize = True)

    # print (morpho_pca)
    # x = pca_data.iloc[:, 0].values
    # y = morpho_pca.transformed.loc[:, ['PC1']].values
    # PLT.scatter(x,y, color = 'r')

    morpho_pca.InvertVectors()
    # print (morpho_pca)

    x = pca_data.iloc[:, 0].values
    y = morpho_pca.transformed.loc[:, ['PC1']].values
    PLT.scatter(x,y, color = 'g')
    PLT.show()


    morpho_pca.Save('data/morphometrics.pca')
    print (morpho_pca)

    component_data = morpho_pca.transformed.loc[:, ['PC1', 'PC2', 'PC3', 'PC4']]
    component_data.columns = ['morpho1', 'morpho2', 'morpho3', 'morpho4']

    # check if conversion is good
    assert NP.allclose(pca_data.values, morpho_pca.ReTraFo(component_data.values))

    # join stride pca
    master_data = master_data.join(component_data, how = 'left')


    # stride parameter PCA
    pca_features = ['distance_diml', 'frequency_diml', 'speed_diml']
    pca_data = master_data.loc[:, pca_features].dropna()
    pca_data = pca_data.astype(float)

    stride_pca = ET.PrincipalComponentAnalysis(pca_data, pca_features, standardize = True)
    stride_pca.Save('data/stride_parameters.pca')
    print (stride_pca)

    component_data = stride_pca.transformed.loc[:, ['PC1', 'PC2', 'PC3']]
    component_data.columns = ['stride1', 'stride2', 'stride3']

    # check if conversion is good
    assert NP.allclose(pca_data.values, stride_pca.ReTraFo(component_data.values))

    # join stride pca
    master_data = master_data.join(component_data, how = 'left')





    ### head angle
    peer_angles = PD.read_csv(config['datafiles']['posture'], sep = ';') \
                   .set_index('cycle_idx', inplace = False) \
                   .loc[:, ['μ_head', 'μ_torso']]
    # head_angle.index = [int(idx[1:]) for idx in head_angle.index.values]
    # print (head_angle)
    peer_angles.columns = ['head_angle', 'torso_angle']

    master_data.index = [f'c{idx:05.0f}' for idx in master_data.index.values]
    master_data.index.name = 'cycle_idx'
    master_data = master_data.join(peer_angles, how = 'left')



    # add extra sex
    sex_extra = PD.read_csv('data/extra_sex.csv', sep = ";").set_index('cycle_idx', inplace = False).loc[:, 'sex'].to_dict()
    master_data.loc[:, 'sex'] = [sex_extra.get(cycle_idx, 'U') if row['sex'] == 'U' else row['sex'] for cycle_idx, row in master_data.iterrows()]

    # add average duty factor
    master_data['dutyfactor'] = NP.nanmean(master_data.loc[:, [f'dutyfactor_{limb}' for limb in ['fl', 'hl']]], axis = 1).ravel()


    # heavy low birth weights are not low birth weight.
    nonnans = NP.logical_not(NP.isnan(NP.array(master_data['LBW'].values, float)))
    # print (NP.nansum(master_data['LBW'].values))
    master_data.loc[nonnans, 'LBW'] = NP.logical_and(master_data.loc[nonnans, 'LBW'].values, master_data.loc[nonnans, 'birth_weight'].values <= 0.8)
    # print (NP.nansum(master_data['LBW'].values))


    ## post hoc data hygiene
    # log transforms
    master_data[f'duty_log'] = NP.log(master_data[f'dutyfactor'].values)
    for limb in ['hl', 'fl']:
        master_data[f'duty_{limb}_log'] = NP.log(master_data[f'dutyfactor_{limb}'].values)
        master_data[f'clr_{limb}_log'] = NP.log(master_data[f'clearance_{limb}'].values)

    # column names
    column_translator = { \
                            'piglet_label': 'piglet' \
                          , 'LBW': 'is_lbw' \
                          , 'iphase': 'phase_hl' \
                          , 'recording_weight': 'weight' \
                          , 'recording_age_h': 'age' \
                         }
    master_data.rename(columns = column_translator, inplace = True)

    ## done!
    print (f'kinematic data extraction done ({master_data.shape[0]} strides)', ' '*16)
    # print(master_data.sample(5).T)
    # master_data.dropna(inplace = True)
    # print(master_data.shape)
    # print (master_data.columns.values)

    # done!
    return master_data



################################################################################
### data assembly                                                            ###
################################################################################

def AssembleData( config \
                  , master: PD.DataFrame = None \
                  , posture: PD.DataFrame = None \
                  , coordination: PD.DataFrame = None \
                 ) -> PD.DataFrame:
    ## assemble data from different sources
    # load meta data
    if master is None:
        master_storage = config['datafiles']['master_data']
        master = PD.read_csv(master_storage, sep = ';').set_index('cycle_idx', inplace = False)

    # load affine components
    if posture is None:
        posture_storage = config['datafiles']['posture']
        posture = PD.read_csv(posture_storage, sep = ';').set_index('cycle_idx', inplace = False)
        posture = posture.loc[:, [col for col in posture.columns if 'φ' not in col]]

    # load coordination pca
    if coordination is None:
        coordination_storage = config['datafiles']['coordination']
        coordination = PD.read_csv(coordination_storage, sep = ';').set_index('cycle_idx', inplace = False)

    ## join all data sources
    data = master.join(posture, how = 'left').join(coordination, how = 'left')

    return data




################################################################################
### Mission Control                                                          ###
################################################################################
if __name__ == "__main__":

    # (B) from kinematics
    master_data = AppendAnalysisData()
    print (master_data)
    master_data.to_csv('../data/master_data.csv', sep = ';')

    # finally, store a combined data table
    # data = AssembleData(config, master = master_extended)
    # data.to_csv(config['datafiles']['analysis_data'], sep = ';')
    # print (f"stored data {data.shape} to {config['datafiles']['analysis_data']}!")
