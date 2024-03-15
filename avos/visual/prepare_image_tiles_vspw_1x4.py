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


def plot_tiles(input_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rows = 1
    cols = 4  # len(img_paths.keys())
    # fig = plt.figure(figsize=(cols*4, rows*3), dpi=100)
    medvt_pred_path = input_paths['MEDVT']
    video_names = sorted(os.listdir(os.path.join(medvt_pred_path)))
    print('total_videos:%d'%len(video_names))
    index = 0
    for video in video_names:
        index += 1
        print('video_no:%d  vid_name:%s' % (index, video))
        seq_out_dir = os.path.join(output_dir, video)
        if not os.path.exists(seq_out_dir):
            os.makedirs(seq_out_dir)
        else:
            continue
        frame_list = sorted([f for f in glob.glob(os.path.join(medvt_pred_path, video) + '/*.png', recursive=False)])
        frame_list = [f.split('/')[-1][:-4] for f in frame_list]

        fig = plt.figure(figsize=(16, 4), dpi=100)
        ax = [plt.subplot(rows, cols, i + 1) for i in range(rows * cols)]
        for a in ax:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_aspect('equal')
            plt.subplots_adjust(wspace=0, hspace=0)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for frame in frame_list:
            idx = 1
            for name, dir_name in input_paths.items():
                if name == 'Image':
                    frame_path = os.path.join(dir_name, video, 'origin', '%s.jpg' % frame)
                else:
                    frame_path = os.path.join(dir_name, video, '%s.png' % frame)
                # import ipdb;ipdb.set_trace()
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(rows, cols, idx)
                plt.axis('off')
                plt.imshow(img)
                plt.title(name, loc='center')
                idx += 1
            plt.subplot_tool()
            f_name = os.path.join(seq_out_dir, '%s.png' % frame)
            plt.savefig(f_name)
        plt.close(fig)
    return


if __name__ == '__main__':
    img_path = '/local/riemann1/home/rezaul/datasets_vss/VSPW_480p/data'
    gt_path = '/local/riemann/home/rezaul/projects/medvt2w/results/segmentation_evals/vss/vspw/medvt_vspw_xx/viz/gt_color_mask'
    medvt_path = '/local/riemann/home/rezaul/projects/medvt2w/results/segmentation_evals/vss/vspw/medvt_vspw_xx/viz/pred_color_mask'
    baseline_path = '/local/riemann/home/rezaul/projects/medvt2w/results/segmentation_evals/vss/vspw/swin_baseline_vspw/viz/pred_color_mask'
    output_dir = '/local/riemann/home/rezaul/projects/medvt2w/results/segmentation_evals/vss/vspw/medvt_vspw_xx/viz/pred_tiles_2'

    input_paths = {
        'Image': img_path,
        'GT': gt_path,
        'Baseline': baseline_path,
        'MEDVT': medvt_path
    }
    plot_tiles(input_paths, output_dir)
