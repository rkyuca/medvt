import numpy as np
import sys
import cv2
import os
from PIL import Image


def read_imgs(imgs_dir):
    imgs = []
    # imgs2 = []
    #    req_sqs = ['drinking_2']
    for root, dirs, files in os.walk(imgs_dir):
        files = sorted(files)
        for f in range(len(files)):
            #            cont = False
            #            for r in req_sqs:
            #                if r in files[f]:
            #                    cont = True
            #            if cont:
            img = cv2.imread(os.path.join(imgs_dir, files[f]))
            # img = cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2))
            imgs.append(img)
    return imgs


def gen_video_back(imgs, vidfile):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(vidfile, fourcc, 5, (imgs[0].shape[1], imgs[0].shape[0]))
    for i in range(len(imgs)):
        video.write(imgs[i])
    cv2.destroyAllWindows()
    video.release()


def gen_video(imgs, vidfile):
    width = imgs[0].shape[1]
    height = imgs[0].shape[0]
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # video = cv2.VideoWriter('%s.avi'%vidfile, fourcc, 15, (imgs[0].shape[1], imgs[0].shape[0]))
    # video = cv2.VideoWriter('%s.avi'%vidfile, 0, 1,  (imgs[0].shape[1], imgs[0].shape[0]))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('%s_10.avi' % vidfile, fourcc, 10, (width, height))

    for i in range(len(imgs)):
        video.write(imgs[i])
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    """
    argv[1]: input images path
    argv[2] : vidfile
    """
    input_dirs = {
        'davis_swinb_vs_rtnet': {
            'path': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/tiles_2x2_swinb_vs_rtnet',
            'seqs': ['breakdance', 'car-roundabout', 'dance-twirl']
        },
        'moca_swinb_vs_rtnet': {
            'path': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/tiles_2x2_swinb_vs_rtnet',
            'seqs': ['flounder_6', 'flatfish_1', 'black_cat_1']
        }
    }

    input_dirs_back = {
        'davis_swinb_vs_baseline': {
            'path': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/tiles_2x2_swinb_vs_baseline',
            'seqs': ['breakdance', 'car-roundabout', 'dance-twirl']
        },
        'moca_swinb_vs_baseline': {
            'path': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/tiles_2x2_swinb_vs_baseline',
            'seqs': ['flounder_6', 'flatfish_1', 'black_cat_1']
        }
    }
    out_dir_root = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/selected_videos'
    for path_key, info in input_dirs.items():
        path = info['path']
        seqs = info['seqs']
        path_token = path.split('/')[-1]
        out_dir = os.path.join(out_dir_root, path_key)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for seq in seqs:
            print('dir_key:%s seq:%s' % (path_key, seq))
            seq_path = os.path.join(path, seq)
            imgs = read_imgs(seq_path)
            outfile = os.path.join(out_dir, '%s' % seq)
            gen_video(imgs, outfile)
