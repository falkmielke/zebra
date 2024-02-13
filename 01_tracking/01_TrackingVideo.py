#!/usr/bin/env python3

import os as OS
import pathlib as PL
import re as RE
import numpy as NP
import pandas as PD
import matplotlib as MP
import matplotlib.pyplot as PLT
from tqdm import tqdm as tqdm

xy = ['x', 'y']
dpi = 150

if False:
    for i in range(3):
        OS.system(f"""ffmpeg -pattern_type glob -i "frames_ep{i+1}/*.png" -c:v libvpx-vp9 -crf 24 ../figures/zebra_ep{i}.webm""")


for i in range(1, 4):
    path = PL.Path(f"frames_ep{i}").glob('**/*')
    files = list(sorted([fi for fi in path if fi.is_file()]))
    # print (files)

    tracking = PD.read_csv(f"../data/tracked_ep{i}.csv", sep = ';')
    #tracking['frame_nr'] += 1
    print (tracking.sample(4).T)

    landmarks = [ \
          "shoulder" \
        , "elbow" \
        , "carpal" \
        , "ankle" \
        ]
    landmarks2 = [ \
          "snout" \
        , "ear" \
        , "withers" \
        , "croup" \
                  ]


    #for fi in NP.random.choice(files, 3):
    for nr, row in tqdm(tracking.iloc[:, :].iterrows()):

        fi = files[nr]
        frame_nr = int(RE.findall(r"[0-9]+", fi.name)[0])
        # print (fi, fi.name)
        # print (row.values)
        # continue

        fig = PLT.Figure(figsize = (1920/dpi, 1080/dpi), dpi = dpi)
        fig.subplots_adjust(bottom = 0., left = 0., top = 1., right = 1.)
        ax = fig.add_subplot(1, 1, 1, aspect = 'equal')

        img = PLT.imread(fi)
        ax.imshow(img, origin = 'upper')

        points = NP.stack([[row[f'{lm}_{coord}'] for coord in xy] for lm in landmarks], axis = 0)
        ax.plot(points[:, 0], points[:, 1], lw = 1, marker = 'o', color = 'darkorange')

        points = NP.stack([[row[f'{lm}_{coord}'] for coord in xy] for lm in landmarks2], axis = 0)
        ax.plot(points[:, 0], points[:, 1], lw = 1, marker = 'o', color = 'lightblue')

        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


        fig.savefig(f'tracked_ep{i}/{frame_nr:04.0f}.png')
        PLT.close()



    if True:
        OS.system(f"""ffmpeg -pattern_type glob -i "tracked_ep{i}/*.png" -c:v libvpx-vp9 -crf 24 ../figures/zebra_ep{i}_tracked.webm""")

"""
snout_x
snout_y
ear_x
ear_y
withers_x
withers_y
croup_x
croup_y
shoulder_x
shoulder_y
elbow_x
elbow_y
carpal_x
carpal_y
ankle_x
ankle_y
hoof_x
hoof_y
"""
