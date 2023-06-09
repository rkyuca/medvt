"""
Davis dataloader with train data augmented by ytvos.
"""
import math
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import logging

from avos.datasets import path_config as dataset_path_config
import avos.datasets.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Davis16TrainDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=6, use_ytvos=False, train_size=300, use_flow=False):
        self.num_frames = num_frames
        self.use_ytvos = use_ytvos
        self.split = 'train'
        self.use_flow = use_flow
        self._transforms = make_train_transform(train_size=train_size)

        self.davis16_train_seqs_file = dataset_path_config.davis16_train_seqs_file
        self.davis16_rgb_path = dataset_path_config.davis16_rgb_path
        self.davis16_gt_path = dataset_path_config.davis16_gt_path
        self.davis16_flow_path = dataset_path_config.davis16_flow_path

        self.ytvos19_rgb_path = dataset_path_config.ytvos19_rgb_path
        self.ytvos19_gt_path = dataset_path_config.ytvos19_gt_path
        self.ytvos19_flow_path = dataset_path_config.ytvos19_flow_path

        self.frames_info = {
            'davis': {},
            'ytvos': {}
        }
        self.img_ids = []
        logger.debug('loading davis 16 train seqs...')
        with open(self.davis16_train_seqs_file, 'r') as f:
            video_names = f.readlines()
            video_names = [name.strip() for name in video_names]
            logger.debug('davis-train num of videos: {}'.format(len(video_names)))
            for video_name in video_names:
                frames = sorted(glob.glob(os.path.join(self.davis16_gt_path, video_name, '*.png')))
                self.frames_info['davis'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
                self.img_ids.extend([('davis', video_name, frame_index) for frame_index in range(len(frames))])
        if self.use_ytvos:
            logger.debug('loading ytvos train seqs...')
            video_names = glob.glob(self.ytvos19_gt_path + "/*/")
            video_names = [video_name.split('/')[-2] for video_name in video_names]
            logger.debug('ytvos-train num of videos: {}'.format(len(video_names)))
            for video_name in video_names:
                frames = sorted(glob.glob(os.path.join(self.ytvos19_gt_path, video_name, '*.png')))
                self.frames_info['ytvos'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
                self.img_ids.extend(
                    [('ytvos', video_name, frame_index) for frame_index in range(0, len(frames), self.num_frames)])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        dataset, video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[dataset][video_name])
        center_frame_name = self.frames_info[dataset][video_name][frame_index]
        frame_indices = [(x + vid_len) % vid_len for x in range(frame_index - math.floor(float(self.num_frames) / 2),
                                                                frame_index + math.ceil(float(self.num_frames) / 2), 1)]
        assert len(frame_indices) == self.num_frames
        frame_ids = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)
        img = []
        masks = []
        mask_paths = []
        flows = []
        skip_current_sample_flow = False
        # import ipdb;ipdb.set_trace()
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)
            if dataset == 'ytvos':
                img_path = os.path.join(self.ytvos19_rgb_path, video_name, frame_name + '.jpg')
                gt_path = os.path.join(self.ytvos19_gt_path, video_name, frame_name + '.png')
                if self.use_flow:
                    prev_frame = '%05d' % (int(frame_name) - 5)
                    next_frame = '%05d' % (int(frame_name) + 5)
                    fwd_flow_file = os.path.join(self.ytvos19_flow_path, video_name,
                                                 prev_frame + '_' + frame_name + '.png')
                    bkd_flow_file = os.path.join(self.ytvos19_flow_path, video_name,
                                                 frame_name + '_' + next_frame + '.png')
            else:
                img_path = os.path.join(self.davis16_rgb_path, video_name, frame_name + '.jpg')
                gt_path = os.path.join(self.davis16_gt_path, video_name, frame_name + '.png')
                if self.use_flow:
                    prev_frame = '%05d' % (int(frame_name) - 1)
                    next_frame = '%05d' % (int(frame_name) + 1)
                    fwd_flow_file = os.path.join(self.davis16_flow_path, video_name,
                                                 prev_frame + '_' + frame_name + '.png')
                    bkd_flow_file = os.path.join(self.davis16_flow_path, video_name,
                                                 frame_name + '_' + next_frame + '.png')
            # import ipdb;ipdb.set_trace()
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255
            masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            if self.use_flow and not skip_current_sample_flow:
                fwd_flow = None
                bkd_flow = None
                if os.path.exists(fwd_flow_file):
                    fwd_flow = Image.open(fwd_flow_file).convert('RGB')
                elif os.path.exists(bkd_flow_file):
                    bkd_flow = Image.open(bkd_flow_file).convert('RGB')
                # TODO fix it later
                # if fwd_flow is not None and bkd_flow is not None:
                #    flow = (fwd_flow + bkd_flow)/2
                # elif fwd_flow is not None:
                if fwd_flow is not None:
                    flow = fwd_flow
                elif bkd_flow is not None:
                    flow = bkd_flow
                else:
                    skip_current_sample_flow=True
                    logger.debug('Flow not found for fwd_flow_file: : '+fwd_flow_file)
                    logger.debug('Flow not found for bkd_flow_file: '+bkd_flow_file)
                    flow = None
                    # raise Exception('Flow file not found for :%s-%s' % (video_name, frame_name))
                flows.append(flow)
            # mask_paths.append(gt_path)
        # import ipdb;ipdb.set_trace()
        masks = torch.cat(masks, dim=0)
        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'mask_paths': mask_paths}
        if self.use_flow and not skip_current_sample_flow:
            target['flows'] = flows
        # import ipdb;ipdb.set_trace()
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # import ipdb;
        # ipdb.set_trace()
        if self.use_flow and not skip_current_sample_flow:
            target['flows'] = [torch.nn.functional.interpolate(target['flows'][i].unsqueeze(0), img[i].shape[-2:]) if target['flows'][i].shape[-2:] !=  img[i].shape[-2:] else target['flows'][i].unsqueeze(0) for i in range(len(img))]
            target['flows'] = torch.cat(target['flows'], dim=0)
        return torch.cat(img, dim=0), target


def make_train_transform(train_size=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([500, 600, 700]),
            T.RandomSizeCrop(473, 750),
            T.RandomResize([train_size], max_size=int(1.8 * train_size)),
        ]),
        normalize,
    ])
