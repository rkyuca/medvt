"""
Custom dataloader for MoCA dataset.
"""
import math
import random
import torch
import torch.utils.data
import os
from PIL import Image
import glob
import logging
import copy
import numpy as np
from numpy import random as rand
import torchvision
import torchvision.transforms

from avos.datasets import path_config as dataset_path_config
import avos.datasets.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MoCAStylized(torch.utils.data.Dataset):

    def __init__(self, num_frames=6, min_size=473, sequence_names=None, perturb='repeat', n_factors=2):
        super(MoCAStylized, self).__init__()
        logger.debug('MoCAStylized----> perturb:%s' % perturb)
        # import ipdb;ipdb.set_trace()
        self.num_frames = num_frames
        self.min_size = min_size
        self.perturb = perturb  # used for interpretability experiments, use 'none' for other cases
        self.dataset_root = dataset_path_config.moca_dataset_root
        self.dataset_path = dataset_path_config.moca_dataset_images_path
        self.moca_flow_path = dataset_path_config.moca_dataset_flow_path
        self.files_list = dataset_path_config.moca_val_set_file
        self._transforms_norm = self.__class__.make_norm_transforms(min_size)
        self.n_factors = n_factors
        self.current_factor = 0
        logger.debug('self.perturb:%s' % self.perturb)
        logger.debug('self.n_factors:%d' % self.n_factors)
        # ###########################################################
        self.styles = ['Lynx', 'Maruska640', 'Zuzka1', 'Zuzka2']  # 'Original',
        self.stylized_moca_path = '/local/riemann1/home/rezaul/datasets_vos/MoCA_Stylized'
        # ###########################################################

        self.frames_info = {}
        self.img_ids = []
        if sequence_names is None:
            with open(self.files_list, 'r') as f:
                video_names = f.readlines()
                video_names = [name.strip() for name in video_names]
        else:
            video_names = sequence_names if isinstance(sequence_names, list) else [sequence_names]
        logger.debug('moca-val num of videos: {}'.format(len(video_names)))
        # video_names = random.sample(video_names, 2)
        # video_names = ['bear', 'flatfish_0', 'flounder_6', 'goat_1','hedgehog_1', 'lion_cub_0', 'octopus_1','seal_1']
        for video_name in video_names:
            frames = sorted(glob.glob(os.path.join(self.dataset_path, video_name, '*.jpg')))
            if len(frames) < 50:
                continue
            self.frames_info[video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
            self.img_ids.extend([(video_name, frame_index) for frame_index in
                                 range(20, len(frames) - 20, 10)])
        logger.debug('selected num of videos: %d' % len(self.frames_info))
        # import ipdb; ipdb.set_trace()
        logger.debug('data loader init: done')

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def make_norm_transforms(min_size=360):
        return T.Compose([
            # T.CenterCrop([min_size], max_size=int(1.8 * min_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __load_gt_paths__(self, p_video_name, p_frame_indices):
        # import ipdb;ipdb.set_trace()
        assert len(p_frame_indices) == self.num_frames
        frame_paths = []
        for frame_id in p_frame_indices:
            frame_name = self.frames_info[p_video_name][frame_id]
            gt_path = os.path.join(self.dataset_root,'Annotations', p_video_name, frame_name + '.png')
            frame_paths.append(gt_path)
        return frame_paths

    def __load_frames__(self, p_video_name, p_frame_indices, style='Original'):
        # import ipdb;ipdb.set_trace()
        frame_indices = []
        frame_paths = []
        frame_list = []
        for frame_id in p_frame_indices:
            frame_name = self.frames_info[p_video_name][frame_id]
            frame_indices.append(frame_name)
            if style == 'Original':
                img_path = os.path.join(self.dataset_path, p_video_name, frame_name + '.jpg')
            else:
                img_path = os.path.join(self.stylized_moca_path, p_video_name, style, frame_name + '.jpg')
            img_i = Image.open(img_path).convert('RGB')
            # img_i = img_i.resize((self.min_size, self.min_size))
            frame_list.append(img_i)
            frame_paths.append(img_path)
        target = {'video_name': p_video_name, 'frame_ids': frame_indices}
        vid_clip, _ = self._transforms_norm(frame_list, target)
        vid_clip = torch.stack(vid_clip, dim=0).permute(1, 0, 2, 3)
        return vid_clip

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[video_name])
        center_frame_name = self.frames_info[video_name][frame_index]
        self.current_factor = (self.current_factor + 1) % self.n_factors
        factor = self.current_factor
        meta_data = {
            'video_name': video_name,
            'center_frame_name': center_frame_name,
        }
        if factor == 0:
            # Same motion different appearance
            styles = random.sample(self.styles, k=2, counts=[1] * 4)
            style1 = styles[0]
            style2 = styles[1]
            # dt = 1 # 5A1
            dt = 1 #  random.sample(population=list(range(10, 20, 1)), k=1)[0]  # default 1random.sample([10, 15, 20], k=1)[0]  # default 1
            frame_indices_1 = [(x + vid_len) % vid_len for x in
                               range(frame_index - dt * math.floor(float(self.num_frames) / 2),
                                     frame_index + dt * math.ceil(float(self.num_frames) / 2), dt)]
            assert len(frame_indices_1) == self.num_frames
            vid_clip_1 = self.__load_frames__(video_name, frame_indices_1, style=style1)
            frame_indices_2 = copy.deepcopy(frame_indices_1)
            vid_clip_2 = self.__load_frames__(video_name, frame_indices_2, style=style2)
            meta_data['style1'] = style1
            meta_data['style2'] = style2
            meta_data['clip1_gt_paths'] = self.__load_gt_paths__(video_name, frame_indices_1)
            meta_data['clip2_gt_paths'] = self.__load_gt_paths__(video_name, frame_indices_2)
            meta_data['frame_indices_1'] = [str(self.frames_info[video_name][index]) for index in frame_indices_1]
            meta_data['frame_indices_2'] = [str(self.frames_info[video_name][index]) for index in frame_indices_2]
        elif factor == 1:
            # same appearance different motion
            ###################################
            style1 = style2 = random.sample(self.styles, k=1)[0]  # 'Original'
            # dt = 10 # random.sample(population=list(range(1, 10, 1)), k=1)[0]  # default 1
            # dt = random.sample(population=list(range(5, 20, 2)), k=1)[0]  # default 1 # 5A1
            dt = random.sample(population=list(range(10, 20, 2)), k=1)[0]  # default 1 # Sal ROC
            # ntc = vid_len // self.num_frames
            # dt = random.sample(list(range(min(10, ntc), min(20, ntc) + 1)), k=1)[0]
            frame_indices_1 = [(x + vid_len) % vid_len for x in
                               range(frame_index - dt * math.floor(float(self.num_frames) / 2),
                                     frame_index + dt * math.ceil(float(self.num_frames) / 2), dt)]
            assert len(frame_indices_1) == self.num_frames
            vid_clip_1 = self.__load_frames__(video_name, frame_indices_1, style=style1)
            if self.perturb == 'shuffle':
                # This is added for temporal order analysis
                fl = [(x + vid_len) % vid_len for x in
                      range(frame_index - math.floor(float(self.num_frames) / 2), frame_index, 1)]
                fr = [(x + vid_len) % vid_len for x in
                      range(frame_index + 1, frame_index + math.ceil(float(self.num_frames) / 2), 1)]
                flr = fl + fr
                random.shuffle(flr)
                frame_indices_2 = flr[:len(fl)] + [frame_index] + flr[len(fl):]
            elif self.perturb == 'repeat':
                # selected_frame_index = random.sample(frame_indices_1, k=1)[0]
                selected_frame_index = frame_index
                frame_indices_2 = [selected_frame_index for _ in range(self.num_frames)]
            elif self.perturb == 'random':
                frame_indices_2 = random.choices(list(range(frame_index)) + list(range(frame_index + 1, vid_len)),
                                                 k=self.num_frames)
                frame_indices_2[len(frame_indices_2) // 2] = frame_index
            else:
                raise NotImplementedError
            assert len(frame_indices_2) == self.num_frames
            vid_clip_2 = self.__load_frames__(video_name, frame_indices_2, style=style2)
            meta_data['style1'] = style1
            meta_data['style2'] = style2
            meta_data['clip1_gt_paths'] = self.__load_gt_paths__(video_name, frame_indices_1)
            meta_data['clip2_gt_paths'] = self.__load_gt_paths__(video_name, frame_indices_2)
            meta_data['frame_indices_1'] = [str(self.frames_info[video_name][index]) for index in frame_indices_1]
            meta_data['frame_indices_2'] = [str(self.frames_info[video_name][index]) for index in frame_indices_2]
        else:
            raise ValueError(f' Factor {factor} is not implemented.')
        return factor, vid_clip_1, vid_clip_2, meta_data

