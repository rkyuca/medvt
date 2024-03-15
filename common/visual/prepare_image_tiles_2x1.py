import sys
import cv2
import os
import glob
import matplotlib.pyplot as plt


def plot_tiles(img_paths, output_dir, video_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = 2
    cols = 1
    fig = plt.figure(figsize=(4, 5), dpi=200)
    ax = [plt.subplot(rows, cols, i + 1) for i in range(rows * cols)]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')
        plt.subplots_adjust(wspace=0, hspace=0)
    # fig.subplots_adjust( wspace=0, hspace=0)
    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)
    for video in video_names:
        print('seq:%s'%video)
        # import ipdb;ipdb.set_trace()
        frame_list = sorted([f for f in glob.glob(os.path.join(img_paths['Input'], video) + '/*.jpg', recursive=False)])
        frame_list = [f.split('/')[-1] for f in frame_list]
        for frame in frame_list:
            idx = 1
            for k, v in img_paths.items():
                img_path = os.path.join(v, video, frame)
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
            'Input': '/local/riemann1/home/rezaul/datasets_vos/MoCA_filtered2/JPEGImages/',
            # 'GT': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/moca/moca_gt_bbox_overlay_red',
            # 'Baseline': 'results/supplement_video/moca/moca_baseline_overlay_red',
            'Ours': 'results/swin_medvt_cvpr/moca/medvt_overlay_red',
        }
        out_dir = 'results/swin_medvt_cvpr/moca/medvt_img_tiles_2x1_moca_2'
        video_names = ['flounder_6']
    elif dataset == 'davis':
        mask_paths = {
            'Input': '/local/riemann/home/rezaul/datasets/DAVIS-data/DAVIS_2016/JPEGImages/480p',
            'GT': '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/davis_gt_overlay_red',
            # 'Baseline': 'results/supplement_video/davis/davis_baseline_overlay_red',
            # 'MED-VT': 'results/supplement_video/davis/davis_med_vt_overlay_red',
        }
        video_names = ['car-roundabout', 'scooter-black']
        out_dir = '/local/riemann/home/rezaul/projects/medvt2-main/results4figures_swin/davis/davis_img_gt_tiles_1x2_v2'
    else:
        raise ValueError('dataset name: %s not implemented' % dataset)
    plot_tiles(mask_paths, out_dir, video_names)

