#!/usr/bin/env python3

import sys as SYS
import numpy as NP
import pandas as PD
SYS.path.append('../toolboxes') # makes the folder where the toolbox files are located accessible to python
import FourierToolbox as FT # Fourier Series toolbox


config = {}
config['fps'] = 25.0
config['landmarks'] = { \
       0: 'pt00' # 'fake landmark for segment angles' \
                      , 99: 'pt99' # 'fake landmark for segment angles' \
                      ,  1: "snout" \
                      ,  2: "ear" \
                      ,  3: "withers" \
                      ,  4: "croup" \
                      ,  5: "shoulder" \
                      ,  6: "elbow" \
                      ,  7: "carpal" \
                      ,  8: "ankle" \
                      ,  9: "hoof" \
                     }


# joints
config['joints'] = { \
                       0: ([4,3,2,1], 'head') \
                    ,  1: ([0,99,3,4], 'torso') \
                    ,  7: ([0,99,5,6], 'shoulder') \
                    ,  8: ([5,6,6,7], 'elbow') \
                    ,  9: ([6,7,7,8], 'wrist') \
                    , 10: ([7,8,8,9], 'metacarpal') \
                    , 11: ([4,3,3,9], 'forelimb') \
                    }

# selection for FCAS
config['joint_selection'] = ['head', 'shoulder', 'elbow', 'wrist', 'metacarpal', 'forelimb']
config['reference_joint'] = 'forelimb'
config['pca_joints'] = ['shoulder', 'elbow', 'wrist', 'metacarpal']

config['fourier_order'] = 8
config['superimposition_choice'] = \
                                 dict( skip_shift = True \
                                     , skip_scale = True \
                                     , skip_rotation = False \
                                   )



def LoadLimbs(cycles) -> dict:
    # load stride cycle data as "limb", i.e. joints associated

    # loop cycles

    # relevant_joints = [config['analysis']['reference_joint']] \
    #                 + eval(config['analysis']['pca_joints']) \
    #                 + ['head']
    relevant_joints = config['joint_selection']

    # ... and store limbs on the way
    limbs = {}
    skipped = []
    count = 0
    for cycle_idx in cycles:

        # data labeling
        label = f'c{cycle_idx:02.0f}'
        print (f'({count}/{len(cycles)})', ' '*16, end = '\r', flush = True)
        count += 1

        # store the results
        fsd_file = f"../data/fsd/{label}.csv"
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


def ReLoadLimbs(cycles, subset: list = None) -> dict:
    # load limbs from storage files

    # loop cycles
    # ... and store limbs on the way
    limbs = {}
    count = 0
    for cycle_idx in cycles:
        if subset is not None:
            if f'c{cycle_idx:05.0f}' not in subset:
                continue

        # data labeling
        label = f'c{cycle_idx:02.0f}'
        print (f'({count}/{len(cycles)})', ' '*16, end = '\r', flush = True)
        count += 1

        # load limb
        limb_file = f"../data/limbs/{label}.lmb"
        limbs[label] = FT.NewLimb.Load(limb_file)

        # if count > 50:
        #     break

    print (f"Limbs loaded ({len(limbs)} cycles)", " "*16)

    return limbs


def StoreLimbs(limbs: dict) -> None:
    # store limbs in a shelf
    for label, lmb in limbs.items():
        limb_file = f"../data/limbs/{label}.lmb"
        lmb.Save(limb_file)


################################################################################
### Helpers                                                                  ###
################################################################################

## Ordered unique elements of a nested list
# https://stackoverflow.com/a/716489
# https://twitter.com/raymondh/status/944125570534621185
CombineNestedList = lambda original: list(dict.fromkeys(sum(original, [])))
