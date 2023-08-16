#!/usr/bin/env python3


################################################################################
### Config Toolbox                                                           ###
################################################################################
__author__      = "Falk Mielke"
__date__        = 20211103

"""
This toolbox contains everything to manage a config file
in the context of the Piglet project.

Questions are welcome (falkmielke.biology@mailbox.org)
"""


################################################################################
### Libraries                                                                ###
################################################################################
#_______________________________________________________________________________
import os as OS             # operating system control and file operations
import re as RE             # regular expressions, to extract patterns from text strings
import configparser as CONF # configuration


#_______________________________________________________________________________
# optionally print progress

StatusText = lambda *txt, **kwargs: None

def SetVerbose():
    # a function to print output of this script from another control script
    global StatusText
    StatusText = lambda *txt, **kwargs: print (*txt, **kwargs)

def SetSilent():
    # a function to mute output of this script from another control script
    global StatusText
    StatusText = lambda *txt, **kwargs: None

# per default: silent operation
SetSilent()


################################################################################
### Settings                                                                 ###
################################################################################
class Config(CONF.ConfigParser):
    # configuration and organization of the kinematics analysis.

    @classmethod
    def Load(cls, load_file: str):
        # read the content of an existing config file
        config = cls()

        # load file
        StatusText(f'loading config from {load_file}.')
        config.read(load_file)

        # remember filename
        config.store_file = load_file

        return config

    @classmethod
    def GenerateDefault(cls, config_filename: str):
        # prepares a settings file using confparser.
        # content are:
        #   - folder structure
        #   - video settings
        #   - analysis parameters


        StatusText('generating default config file.')
        config = cls()

        config.store_file = config_filename

        # folders to config
        config['folders'] = { \
                              fld: f'data{OS.sep}{fld}' \
                              for fld in [  'kinematics' \
                                            , 'jointangles' \
                                            , 'jointfsd' \
                                            , 'limbs' \
                                          ] \
                             }

        # video settings
        config['video'] = { \
                              'resolution_x': 1920.0 \
                            , 'resolution_y': 500.0 \
                            , 'framerate': 50. \
                            , 'chessboard_ref_width': 0.375 \
                           }


        # files
        config['datafiles'] = { \
                                 'raw_folder': f'data{OS.sep}kinematics' \
                               , 'master_data': f'data{OS.sep}master_data.csv' \
                               , 'exclusion': f'data{OS.sep}cycle_exclusion.csv' \
                               , 'subject_info': f'data{OS.sep}master_animals.csv' \
                               , 'kinematics_file': f'data{OS.sep}all_strides.h5' \
                               , 'spatialcalibration': f'data{OS.sep}scaling_chessboards.csv' \
                               , 'stridetimes_file': f'data{OS.sep}stride_timing.csv' \
                               , 'jointangles_file': f'data{OS.sep}stride_cyclediffs.csv' \
                               , 'cyclediffs_file': f'data{OS.sep}stride_cyclediffs.csv' \
                               , 'cyclemissing_file': f'data{OS.sep}stride_missing.csv' \
                               , 'coordination_data': f'data{OS.sep}coordination_data.csv'  \
                               , 'coordination_pca': f'data{OS.sep}coordination.pca'  \
                               , 'posture': f'data{OS.sep}analysis_posture.csv' \
                               , 'coordination': f'data{OS.sep}analysis_coordination.csv'  \
                               , 'analysis_data': f'data{OS.sep}analysis_all.csv' \
                              }

        # analysis parameters
        config['analysis'] = { \
                                 'fourier_order': 8 \
                               , 'flip_rightwards': True \
                               , 'reference_joint': 'forelimb' \
                               , 'superimposition_choice': \
                                            dict( skip_shift = True \
                                                , skip_scale = True \
                                                , skip_rotation = False \
                                              ) \
                               , 'pca_joints': [ \
                                                   'shoulder', 'elbow', 'wrist' \
                                                 , 'hip', 'knee', 'ankle' \
                                                ] \
                               , 'toe_landmark': 17 \
                               , 'toemove_threshold': 1.0 \
                               , 'trunk_landmarks': [4,5] \
                               # , 'fl_clearance_landmarks': [4,16] \
                               # , 'hl_clearance_landmarks': [5,10] \
                               , 'fl_clearance_landmarks': [13,16] \
                               , 'hl_clearance_landmarks': [7,10] \
                               # , 'clearance_trafo': 'lambda x: NP.log(x+1.)' \
                               , 'speedref_landmarks': [4,5,6] \
                              }

        # landmarks
        config['landmarks'] = { \
                                 0: 'pt00' # 'fake landmark for segment angles' \
                              , 99: 'pt99' # 'fake landmark for segment angles' \
                              ,  1: 'snout' \
                              ,  2: 'eye' \
                              ,  3: 'ear' \
                              ,  4: 'withers' \
                              ,  5: 'croup' \
                              ,  6: 'tailbase' \
                              ,  7: 'hip' \
                              ,  8: 'knee' \
                              ,  9: 'ankle' \
                              , 10: 'metatarsal' \
                              , 11: 'hhoof' \
                              , 12: 'scapula' \
                              , 13: 'shoulder' \
                              , 14: 'elbow' \
                              , 15: 'wrist' \
                              , 16: 'metacarpal' \
                              , 17: 'fhoof' \
                              }

        # joints
        config['joints'] = { \
                               0: ([5,4,2,1], 'head') \
                            ,  1: ([0,99,4,5], 'torso') \
                            ,  2: ([4,5,7,8], 'hip') \
                            ,  3: ([7,8,8,9], 'knee') \
                            ,  4: ([8,9,9,10], 'ankle') \
                            ,  5: ([9,10,10,11], 'metatarsal') \
                            ,  6: ([5,4,12,13], 'scapula') \
                            ,  7: ([12,13,13,14], 'shoulder') \
                            ,  8: ([13,14,14,15], 'elbow') \
                            ,  9: ([14,15,15,16], 'wrist') \
                            , 10: ([15,16,16,17], 'metacarpal') \
                            , 11: ([5,4,13,16], 'forelimb') \
                            , 12: ([4,5,7,10], 'hindlimb') \
                            }

        # subset of joints that enter the analysis
        config['selection'] = {nr: jnt for nr, jnt in enumerate([ \
                                  'hip' \
                                , 'knee' \
                                , 'ankle' \
                                , 'metatarsal' \
                                , 'scapula' \
                                , 'shoulder' \
                                , 'elbow' \
                                , 'wrist' \
                                , 'metacarpal' \
                                , 'torso' \
                                , 'head' \
                                , 'forelimb' \
                                , 'hindlimb' \
                               ]) }


        # excluded cycles
        config['exclude'] = {**{ cycle_idx: 'no forelimb' \
                              for cycle_idx in ['2765', '2835', '1905']} \
                             , **{ cycle_idx: 'joint missing' \
                              for cycle_idx in ['2401', '3477', '3501', '3702', '3766', '3791', '3908']} \
                             , **{ cycle_idx: 'double hindlimb touchdown' \
                              for cycle_idx in ['2613']} \
                             , **{ cycle_idx: 'joint angles off' \
                              for cycle_idx in ['2405', '3432', '1569']} \
                             , **{ cycle_idx: 'missing l_animal' \
                              for cycle_idx in ['2460', '2766']} \
                             , **{ cycle_idx: 'metatarsal off' \
                              for cycle_idx in ['2528', '4049', '1806']} \
                             , **{ cycle_idx: 'shoulder off' \
                              for cycle_idx in ['3481']} \
                             }


        # store config
        config.Store()

        return config

    def LandmarkTranslator(self):
        # translate landmark to number
        return {v: k for k,v in self['landmarks'].items()}

    def Store(self, store_file: str = None, overwrite: bool = False):
        # store config, optionally to a different file.

        if store_file is None:
            store_file = self.store_file

        # check if file exists
        if OS.path.exists(store_file) and not overwrite:

            while ( res:=input(f"Overwrite {store_file}? (y/n)").lower() ) not in {"y", "n"}: pass
            if not res == 'y':
                StatusText('not saved!')
                return

        # write the file
        with open(store_file, 'w') as configfile:
            self.write(configfile)
            StatusText(f'\tstored to {store_file}.')


    def CreateFolders(self):
        # create empty folders
        for fld in self['folders'].values():

            if not OS.path.exists(fld):
                StatusText(f'\tcreating folder: {fld}')
                OS.mkdir(fld)


################################################################################
### Running Example                                                          ###
################################################################################
def ExampleConfig():
    # display output
    SetVerbose()

    # give a path
    config_file = OS.sep.join(['..', 'data','piglets.conf'])

    # generate a project file
    config = Config.GenerateDefault(config_file)

    # load a project file
    config = Config.Load(config_file)

    print(config)
