#!/usr/bin/env python3

import os as OS
import skimage.io as IO
import skimage.registration as REG
import skimage.color as COL
import pathlib as PL
import matplotlib.pyplot as PLT
import numpy as NP
import pandas as PD

# https://scikit-image.org/docs/stable/auto_examples/registration/plot_opticalflow.html

speeds = {}
idx = 0
for episode in [1, 2, 3]:
    basedir = PL.Path(f'../01_tracking/frames_ep{episode}')
    all_frames = list(sorted(OS.listdir(basedir)))
    frame1 = frame2 = None

    for frame_nr, frame_new in enumerate(all_frames):
        if frame1 is None:
            frame1 = frame_new
            img1 = COL.rgb2gray(IO.imread(basedir/frame1))
            continue

        frame2 = frame1
        frame1 = frame_new
        img2 = img1
        img1 = COL.rgb2gray(IO.imread(basedir/frame1))

        y = 900

        _, u = REG.optical_flow_tvl1(img1, img2)
        mean_flow = NP.mean(u[880:920, :].ravel())
        print (episode, frame_nr, mean_flow)

        speeds[idx] = {'ep': int(episode), 'fr': int(frame_nr), 'flow': mean_flow}
        idx += 1

        # phaseshift = REG.phase_cross_correlation(img1, img2)
        # v, u = REG.optical_flow_tvl1(img1, img2)
        # norm = NP.sqrt(u ** 2)
        # NP.mean(u[850:950, :], axis = 1)
        # print (episode, frame_nr)
        # PLT.imshow(u[850:950, :])
        # PLT.show()
        # pause

        # fig, (ax) = PLT.subplots(1, 1, figsize=(8, 8))

        # # --- Quiver plot arguments

        # nvec = 32  # Number of vectors to be displayed along each image dimension
        # nl, nc = img1.shape
        # step = max(nl//nvec, nc//nvec)

        # y, x = NP.mgrid[:nl:step, :nc:step]
        # u_ = u[::step, ::step]
        # v_ = 0*v[::step, ::step]

        # ax.imshow(norm)
        # ax.quiver(x, y, u_, v_, color='r', units='dots',
        #            angles='xy', scale_units='xy', lw=3)
        # ax.set_title("Optical flow magnitude and vector field")
        # ax.set_axis_off()
        # fig.tight_layout()

        # PLT.show()

        # pause

    speeds = PD.DataFrame.from_dict(speeds).T
    print(speeds)
    speeds.to_csv(f'optical_flow_ep{episode}.csv', sep = ';')
