#!/usr/bin/env python3

"""
common I/O functions
and little helpers
"""

################################################################################
### Libraries                                                                ###
################################################################################
import sys as SYS           # system control
import os as OS             # operating system control and file operations
import numpy as NP          # numerical analysis
import pandas as PD         # data management
import scipy.interpolate as INTP # interpolation

import ConfigToolbox as CONF # project configuration
import FourierToolbox as FT # Fourier Series toolbox


################################################################################
### Data I/O                                                                 ###
################################################################################

# create the usual file label from a data frame row label
LabelForRow = lambda cycle_idx, row: f"""{row['session_idx']} {row['recording_idx']} c{cycle_idx:05.0f}"""

def LoadSubjectInfo(config: CONF.Config) -> PD.DataFrame:
    # gather info of the subjects
    # Note: is already in master file (LoadMasterData)
    # but might be useful for subject parameter model

    subject_info = PD.read_csv(config['datafiles']['subject_info'], sep = ';')
    subject_info.set_index('piglet_label', inplace = True)
    # print (subject_info)

    return subject_info



def LoadMasterData(config: CONF.Config) -> PD.DataFrame:
    # load stride cycle master data

    ### load all cycles
    master_data = PD.read_csv(config['datafiles']['master_data'], sep = ';')
    master_data.set_index('cycle_idx', inplace = True, drop = True)
    master_data.drop(columns = ['Unnamed: 0'], inplace = True)

    ### excluded due to auto measures
    the_excluded = [str(excl).lower() == 'nan' for excl in master_data['exclude'].values]
    # print ('remainder:', sum(the_excluded), master_data.shape[0])
    master_data = master_data.loc[the_excluded, :]
    master_data.drop(columns = ['exclude'], inplace = True)


    ### exclude due to manual inspection
    exclusion = PD.read_csv(config['datafiles']['exclusion'], sep = ';')
    exclusion.set_index('cycle_idx', inplace = True)

    # # optionally leave in "dlc" tags for jana
    # exclusion = exclusion.loc[NP.logical_not(exclusion['exclusion'] == 'digitization'), :]
    # exclusion = exclusion.loc[NP.logical_not(exclusion['exclusion'] == 'dlc'), :]

    master_data = master_data.join(exclusion, how = 'left')

    the_included = [str(excl).lower() == 'nan' for excl in master_data['exclusion'].values]
    # print ('remainder:', sum(the_excluded), master_data.shape[0])
    master_data = master_data.loc[the_included, :]
    master_data.drop(columns = ['exclusion'], inplace = True)


    ### exclude for other reasons
    the_excluded = list(map(int, config['exclude'].keys()))
    master_data = master_data.loc[[idx for idx in master_data.index.values \
                                   if idx not in the_excluded] \
                                  , :]

    ### return
    # print (master_data)
    # print (master_data.sample(3).T)
    return master_data




################################################################################
### Limbs                                                                    ###
################################################################################

def LoadJoints(config) -> dict:
    # convert joint dictionary from a config (loaded as string)
    # to the desired joint definitions
    return {int(key): eval(value) for key, value in config['joints'].items()}


def LoadLimbs(config) -> dict:
    # load stride cycle data as "limb", i.e. joints associated

    # loop cycles
    master_data = LoadMasterData(config)
    # master_data = master_data.iloc[:64, :]

    # check excluded cycles
    excluded = list(map(int, config['exclude'].keys()))
    master_data = master_data.loc[[idx for idx in master_data.index.values \
                                   if idx not in excluded] \
                                  , :]

    # relevant_joints = [config['analysis']['reference_joint']] \
    #                 + eval(config['analysis']['pca_joints']) \
    #                 + ['head']
    relevant_joints = [v for v in config['selection'].values()]

    # ... and store limbs on the way
    limbs = {}
    skipped = []
    count = 0
    for cycle_idx, master_row in master_data.iterrows():

        # data labeling
        label = f'c{cycle_idx:05.0f}'
        print (f'({count}/{master_data.shape[0]})', LabelForRow(cycle_idx, master_row), ' '*16, end = '\r', flush = True)
        count += 1

        # store the results
        fsd_file = f"{config['folders']['jointfsd']}{OS.sep}c{cycle_idx:05.0f}.csv"
        coefficient_df = PD.read_csv(fsd_file, sep = ';').set_index(['re_im', 'coeff'], inplace = False)

        # create a limb
        lmb = FT.NewLimb(coefficient_df, label = label, coupled = True)

        if NP.any([jnt not in lmb.keys() for jnt in relevant_joints]):
            skipped.append(label)
            continue


        # lmb.PrepareRelativeAlignment(test_joint, skip_rotate = True, skip_center = False, skip_normalize = False)
        limbs[label] = lmb

    print (f"Limbs loaded ({len(limbs)} cycles)", " "*16)
    print ('skipped', skipped)

    return limbs


def ReLoadLimbs(config: CONF.Config, subset: list = None) -> dict:
    # load limbs from storage files

    # loop cycles
    master_data = LoadMasterData(config)

    # check excluded cycles
    excluded = list(map(int, config['exclude'].keys()))
    master_data = master_data.loc[[idx for idx in master_data.index.values \
                                   if idx not in excluded] \
                                  , :]


    # ... and store limbs on the way
    limbs = {}
    count = 0
    for cycle_idx, master_row in master_data.iterrows():
        if subset is not None:
            if f'c{cycle_idx:05.0f}' not in subset:
                continue

        # data labeling
        label = f'c{cycle_idx:05.0f}'
        print (f'({count}/{master_data.shape[0]})', LabelForRow(cycle_idx, master_row), ' '*16, end = '\r', flush = True)
        count += 1

        # load limb
        limb_file = f"{config['folders']['limbs']}{OS.sep}{label}_{config['analysis']['reference_joint']}.lmb"
        limbs[label] = FT.NewLimb.Load(limb_file)

        # if count > 50:
        #     break

    print (f"Limbs loaded ({len(limbs)} cycles)", " "*16)

    return limbs


def StoreLimbs(config, limbs: dict) -> None:
    # store limbs in a shelf
    for label, lmb in limbs.items():
        limb_file = f"{config['folders']['limbs']}{OS.sep}{label}_{config['analysis']['reference_joint']}.lmb"

        lmb.Save(limb_file)


################################################################################
### Analysis Data                                                            ###
################################################################################
# transformations
LogTraFo = lambda vec: NP.log(vec)
UnLogTraFo = lambda lvec: NP.log(lvec)
Center = lambda vec: vec - NP.mean(vec)

# data i/o
def LoadAnalysisData(config: CONF.Config) -> PD.DataFrame:

    # load data
    data = PD.read_csv(config['datafiles']['analysis_data'], sep = ';') \
             .set_index('cycle_idx', inplace = False)

    # make categoricals
    # (i) explicit categories
    data['ageclass'] = PD.Categorical(data['ageclass'].values \
                                 , ordered = True \
                                 )

    # (ii) default categories
    for param in ['piglet', 'litter', 'sex']:
        data[param] = PD.Categorical(data[param].values \
                                 , ordered = True \
                                 )

    # (iii) booleans
    for cat, reference_value in { \
                                  'sex': 'female' \
                                , 'ageclass': 'p10' \
                                , 'is_lbw': False \
                                 }.items():
        for val in NP.unique(data[cat].values):
            # skip the reference value
            if val == reference_value:
                continue

            # create a boolean
            data[f'{cat}_is_{val}'] = NP.array(data[cat].values == val, dtype = float)

    ## logarithmize body proportions
    # for param in [  'bodymass' \
    #               , 'leg_m' \
    #               , 'bmi' \
    #               ]:
    #     data[f'l{param}'] = LogTraFo(data[param].values)

    ## centered (population)
    for param in [ \
                      'duty_fl_log' \
                    , 'clr_fl_log' \
                    , 'duty_hl_log' \
                    , 'clr_hl_log' \
                  ]:
        data[f'{param}_c'] = data[param].values - NP.nanmean(data[param].values)

    ## centered by ageclass
    for param in [  'weight' \
                  , 'morpho1' \
                  ]:
        data[f'{param}_cac'] = NP.nan # initialize empty

        # ... should be groupwise
        for agegrp in NP.unique(data['ageclass'].values):
            selection = data['ageclass'].values == agegrp

            # center
            data.loc[selection, f'{param}_cac'] = Center(data.loc[selection, param].values)

    # log trafo
    for param in [  'weight' \
                  , 'age' \
                  ]:
        data[f'{param}_log'] = NP.log(data[f'{param}'].values)
        # Note: "morpho1" is a PCA result and needs not be logged.
        # Note: some "weight"s were not recorded

    # exclude those with unknown birth weight
    data = data.loc[NP.logical_not(data['is_lbw_is_nan'].values), :]
    # exclude those with unknown sex
    data = data.loc[NP.logical_not(data['sex_is_U'].values), :]

    return data

################################################################################
### Helpers                                                                  ###
################################################################################

## Ordered unique elements of a nested list
# https://stackoverflow.com/a/716489
# https://twitter.com/raymondh/status/944125570534621185
CombineNestedList = lambda original: list(dict.fromkeys(sum(original, [])))


## add a zero column to allow 3D magic
Fake3D = lambda points: NP.c_[ points, NP.zeros((points.shape[0], 1)) ]


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
### Data Access Convenience                                                  ###
################################################################################
def LoadDefaultConfig() -> CONF.Config:
    # load the default config file

    # give config path
    config_file = f'data{OS.sep}piglets.conf'

    # load config file
    config = CONF.Config.Load(config_file)

    return config
