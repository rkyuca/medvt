import argparse
import logging
# from PIL import Image
import random
import socket
import sys

import numpy as np
import cv2
import ast
import operator
import csv
import pandas as pd
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_tiles(img_paths, output_dir, video_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    rows = 2
    cols = 2  # len(img_paths.keys())
    # fig = plt.figure(figsize=(cols*4, rows*3), dpi=100)

    fig = plt.figure(figsize=(8, 6), dpi=100)

    ax = [plt.subplot(rows, cols, i + 1) for i in range(rows * cols)]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')
        plt.subplots_adjust(wspace=0, hspace=0)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    for video in video_names:
        print('seq:%s'%video)
        frame_list = sorted(
            [f for f in glob.glob(os.path.join(list(img_paths.values())[1], video) + '/*.jpg', recursive=False)])
        frame_list = [f.split('/')[-1] for f in frame_list]
        for frame in frame_list:
            idx = 1
            for k, v in img_paths.items():
                img_path = os.path.join(v, video, frame)
                # import ipdb; ipdb.set_trace()
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(rows, cols, idx)
                plt.axis('off')
                plt.imshow(img)
                plt.title(k, loc='center')
                idx += 1
            plt.subplot_tool()
            seq_out_dir = os.path.join(output_dir, video)
            if not os.path.exists(seq_out_dir):
                os.makedirs(seq_out_dir)
            f_name = os.path.join(seq_out_dir, '%s.jpeg' % frame[:-4])
            plt.savefig(f_name)
    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Dataset name???')
    dataset = sys.argv[1]
    assert dataset in ['davis', 'moca']
    print('plotting dataset:%s' % dataset)

    if dataset == 'moca':
        mask_paths = {
            'Input': '/local/riemann/home/msiam/MoCA_filtered2/JPEGImages/',
            'GT': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_gt_bbox_overlay_red',
            # 'Baseline': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_baseline_r101ssss_overlay_red',
            'RTNet': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_rtnet_overlay_red',
            'Ours': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_swinb_ssms_overlay_red',
        }
        out_dir = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/tiles_2x2_swinb_vs_rtnet'
        video_names =  ['flounder_6', 'flatfish_1', 'black_cat_1']
    elif dataset == 'davis':
        mask_paths = {
            'Input': '/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/JPEGImages/480p',
            'GT': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/davis_gt_overlay_red',
            # 'Baseline': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/davis_baseline_r101ssss_overlay_red',
            'RTNet' : '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/davis_rtnet_overlay_red',
            'Ours': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/swinb_ssms_crf_overlay_red',
        }
        out_dir = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/tiles_2x2_swinb_vs_rtnet'
        video_names = ['paragliding-launch']  # ['scooter-black']  # ['breakdance', 'car-roundabout', 'dance-twirl']
    else:
        raise ValueError('dataset name: %s not implemented' % dataset)
    plot_tiles(mask_paths, out_dir, video_names)

