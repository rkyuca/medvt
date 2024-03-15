
import sys
import os
sys.path.append('./')
import cv2
from common.visual.overlay import create_overlay
import argparse

parser = argparse.ArgumentParser()
# Backbone
parser.add_argument('--dataset', default='ms3', type=str,help="")
parser.add_argument("--mask_type", default="all", type=str, help="")
parser.add_argument("--img_ext", default='png', type=str)
parser.add_argument('--mask_ext', default='png', type=str, help="")
parser.add_argument("--gt_ext", default='png', type=str)
args = parser.parse_args()


def save_gt_overlays_avs(main_dir, mask_dir, seq_names, overlay_dir, img_ext, mask_ext):
    for d in seq_names:
        if not os.path.exists(os.path.join(mask_dir, d)):
            print('mask dir not exists: %s' % str(os.path.join(mask_dir, d)))
            continue
        print('current sequence:%s' % d)
        if not os.path.exists(os.path.join(overlay_dir, d)):
            os.makedirs(os.path.join(overlay_dir, d))
        # import ipdb;ipdb.set_trace()
        files = sorted(os.listdir(os.path.join(main_dir, d)))
        for i in range(len(files)):
            if files[i].split('.')[-1] != img_ext:
                print(f'given ext not matched, skip file {files[i]}')
                continue
            print(files[i])
            img = cv2.imread(os.path.join(main_dir, d, files[i]))
            # import ipdb;ipdb.set_trace()
            mask_file = os.path.join(mask_dir, d, files[i].replace('.mp4',''))
            if not os.path.exists(mask_file):
                mask_file = os.path.join(mask_dir, d, '%s_%s.png' % (d, files[i].split('.')[0]))  # for rtnet files
            mask2 = cv2.imread(mask_file, 0)
            if mask2 is None:
                print('mask not found for %s' % d)
                continue
            if mask2.shape[0] != img.shape[0]:
                img = cv2.resize(img, mask2.shape[::-1])
            # mport ipdb; ipdb.set_trace()
            if mask2.max() > 1:
                mask2[mask2 > 100] = 255
                mask2[mask2 <= 100 ] = 0
            overlay = create_overlay(img, mask2, [0, 255])
            cv2.imwrite(os.path.join(overlay_dir, d, files[i]), overlay)


def save_pred_overlays_avs(main_dir, mask_dir, seq_names, overlay_dir, img_ext, mask_ext):
    for d in seq_names:
        if not os.path.exists(os.path.join(mask_dir, d)):
            print('mask dir not exists: %s' % str(os.path.join(mask_dir, d)))
            continue
        print('current sequence:%s' % d)
        if not os.path.exists(os.path.join(overlay_dir, d)):
            os.makedirs(os.path.join(overlay_dir, d))
        # import ipdb;ipdb.set_trace()
        files = sorted(os.listdir(os.path.join(main_dir, d)))
        for i in range(len(files)):
            if files[i].split('.')[-1] != img_ext:
                print(f'given ext not matched, skip file {files[i]}')
                continue
            print(files[i])
            img = cv2.imread(os.path.join(main_dir, d, files[i]))
            # import ipdb;ipdb.set_trace()
            # Fix this in the save later
            mf = files[i].replace('.mp4','')
            mf = '_'.join(mf.split('.')[0].split('_')[:-1])+'_'+str(int(mf.split('.')[0].split('_')[-1]) - 1)+'.png'
            # pred_mask_file_name.split()
            mask_file = os.path.join(mask_dir, d, mf)
            if not os.path.exists(mask_file):
                mask_file = os.path.join(mask_dir, d, '%s_%s.png' % (d, files[i].split('.')[0]))  # for rtnet files
            mask2 = cv2.imread(mask_file, 0)
            if mask2 is None:
                print('mask not found for %s' % d)
                continue
            if mask2.shape[0] != img.shape[0]:
                img = cv2.resize(img, mask2.shape[::-1])
            # mport ipdb; ipdb.set_trace()
            if mask2.max() > 1:
                mask2[mask2 > 100] = 255
                mask2[mask2 <= 100 ] = 0
            overlay = create_overlay(img, mask2, [0, 255])
            cv2.imwrite(os.path.join(overlay_dir, d, files[i]), overlay)


def main(args):

    if args.dataset == 's4':
        main_dir = '/local/riemann/home/rezaul/datasets/avsbench_data/Multi-sources/ms3_data/visual_frames'
        gt_dir = '/local/riemann/home/rezaul/datasets/avsbench_data/Multi-sources/ms3_data/gt_masks/test'
        mask_dir = '/local/riemann/home/rezaul/outputs/avsbench/avs_ms3/test_logs/ms3_press4_20240106-072633_miou_666_fscore755/pred_masks'
        out_dir = '/local/riemann/home/rezaul/outputs/avsbench/avs_ms3/test_logs/ms3_press4_20240106-072633_miou_666_fscore755'
        seq_names = sorted(os.listdir(mask_dir))
    elif args.dataset == 'ms3':
        main_dir = '/local/riemann/home/rezaul/datasets/avsbench_data/Multi-sources/ms3_data/visual_frames'
        gt_dir = '/local/riemann/home/rezaul/datasets/avsbench_data/Multi-sources/ms3_data/gt_masks/test'
        mask_dir = '/local/riemann/home/rezaul/outputs/avsbench/avs_ms3/test_logs/ms3_pvt_20231228-182759_miou613_fscore693/pred_masks'
        out_dir = '/local/riemann/home/rezaul/outputs/avsbench/avs_ms3/test_logs/ms3_pvt_20231228-182759_miou613_fscore693'
        seq_names = sorted(os.listdir(mask_dir))
    else:
        exit()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.mask_type in ['gt','all']:
        overlay_dir = os.path.join(out_dir, 'gt_overlay')
        os.makedirs(overlay_dir,exist_ok=True)
        save_gt_overlays_avs(main_dir, gt_dir, seq_names, overlay_dir, args.img_ext, args.gt_ext )

    if args.mask_type in ['pred','all']:
        overlay_dir = os.path.join(out_dir, 'pred_overlay')
        os.makedirs(overlay_dir,exist_ok=True)
        save_pred_overlays_avs(main_dir, mask_dir, seq_names, overlay_dir, args.img_ext, args.mask_ext)


main(args)
