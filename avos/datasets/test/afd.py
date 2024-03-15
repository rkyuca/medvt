"""
Davis dataloader for inference.
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
import copy

from avos.datasets import path_config as dataset_path_config
import avos.datasets.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AFDDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=6, val_size=473, sequence_names=None, use_flow=False, max_sc=None):
        super(AFDDataset, self).__init__()
        # print('Davis16ValDataset->use_flow:'+str(use_flow))
        self.num_frames = num_frames
        self.split = 'val'
        self.im_size = val_size
        self._transforms = make_validation_transforms()

        self.data_path = '/local/riemann1/home/rezaul/Rezaul_AFD'
        self.img_ids = []
        # afd_jumpingjack_seed_flounder
        # afd_hoola_seed_flounder
        afd_fl = sorted(glob.glob(os.path.join(self.data_path, 'afd_jumpingjack_seed_flounder', '*.png')))
        frames = sorted(glob.glob(os.path.join(self.data_path, 'raw', '*.png')))
        afd = sorted(glob.glob(os.path.join(self.data_path, 'afd', '*.png')))
        flow = sorted(glob.glob(os.path.join(self.data_path, 'flow', '*.png')))
        sparse = sorted(glob.glob(os.path.join(self.data_path, 'greysparse', '*.png')))
        bright = sorted(glob.glob(os.path.join(self.data_path, 'bright', '*.png')))
        self.data = {
            'afd_fl':afd_fl,
            'raw': frames,
            'afd': afd,
            'flow': flow,
            'bright': bright,
            'sparse': sparse,
        }
        self.img_ids = self.data['afd_fl']

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid_len = len(self.img_ids)
        frame_indices = [(x + vid_len) % vid_len for x in range(idx - math.floor(float(self.num_frames) / 2),
                                                                idx + math.ceil(float(self.num_frames) / 2), 1)]
        assert len(frame_indices) == self.num_frames
        data = {
        }
        cats = ['afd_fl']
        images = []
        for kk in cats:  # , vv in self.data.items():
            vv = self.data[kk]
            data[kk] = []
            for fid in frame_indices:
                img = Image.open(vv[fid]).convert('RGB')
                data[kk].append(img)
                if kk == 'afd_fl':
                    images.append(Image.open(vv[fid]).convert('RGB'))
        # import ipdb;ipdb.set_trace()
        """
        data['afd_trans'] = []
        for i in range(len(frame_indices)):
            im = np.asarray(data['raw'][i])
            h,w,_ = im.shape
            afd = np.asarray(data['afd_fl'][i])
            afd_mean = np.tile(afd.mean(axis=(0, 1))[np.newaxis, np.newaxis, :], (h, w, 1))
            afd_std = np.tile(afd.std(axis=(0, 1))[np.newaxis, np.newaxis, :], (h, w, 1))
            im_mean = np.tile(im.mean(axis=(0, 1))[np.newaxis, np.newaxis, :], (h, w, 1))
            im_std = np.tile(im.std(axis=(0, 1))[np.newaxis, np.newaxis, :], (h, w, 1))
            # import ipdb; ipdb.set_trace()
            afd = cv2.resize(afd, (w,h))
            afd2 = np.nan_to_num(np.nan_to_num((afd-afd_mean)/afd_std)*im_std+im_mean)
            afd2[afd2>255]=255
            afd2[afd2<0]=0

            afd2 = Image.fromarray(afd2.astype(np.uint8))
            data['afd_trans'].append(afd2)
        """
        ########################
        # data['afdi']  = [Image.fromarray(np.uint8(0.5*np.array(afd)+0.5*np.array(raw))) for afd,raw in zip(data['afd'], data['raw']) ]
        # data['afdif']  = [Image.fromarray(np.uint8(0.8*np.array(afd)+0.2*np.array(raw))) for afd,raw in zip(data['afd'], data['raw']) ]
        ##############################
        target = {'dataset': 'afd', 'frame_ids': frame_indices, 'vid_len': vid_len}
        for k, v in data.items():
            td, tt = self._transforms(v, target)
            v = td
            # import ipdb;ipdb.set_trace()
            data[k] = torch.stack(v, dim=0)
        target['data'] = data
        images, _ = self._transforms(images, target)
        # import ipdb;ipdb.set_trace()
        return torch.cat(images, dim=0), target


def make_validation_transforms(min_size=360, max_sc=None):
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])








