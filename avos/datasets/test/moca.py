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


class MoCADataset(torch.utils.data.Dataset):

    def __init__(self, num_frames=6, min_size=473, sequence_names=None, use_flow=False, perturb='none',
                 sampling_stride=1):
        super(MoCADataset, self).__init__()

        # import ipdb;ipdb.set_trace()
        self.num_frames = num_frames
        self.use_flow = use_flow
        self.min_size = min_size
        self.perturb = perturb  # used for interpretability experiments, use 'none' for other cases
        self.dataset_path = dataset_path_config.moca_dataset_images_path
        self.moca_flow_path = dataset_path_config.moca_dataset_flow_path
        self.files_list = dataset_path_config.moca_val_set_file
        self._transforms = make_validation_transforms(min_size=min_size)

        self.frames_info = {}
        self.img_ids = []
        # import ipdb;ipdb.set_trace()
        if sequence_names is None:
            with open(self.files_list, 'r') as f:
                video_names = f.readlines()
                video_names = [name.strip() for name in video_names]
        else:
            video_names = sequence_names if isinstance(sequence_names, list) else [sequence_names]
        logger.debug('moca-val num of videos: {}'.format(len(video_names)))
        for video_name in video_names:
            frames = sorted(glob.glob(os.path.join(self.dataset_path, video_name, '*.jpg')))
            ######################
            begin, end, step = 0, len(frames), 1
            if sampling_stride > 1:
                step = sampling_stride
            frames = frames[begin:end:step]
            # This part is for interpretability experiment on 3 sampled frames
            # begin = len(frames) // 4
            # step = len(frames) // 4
            # end = (3 * len(frames)) // 4 + 1
            # frames = frames[begin:end:step]
            # print('video_name:%s ----------- num_frames: %d' % (video_name, len(frames)))
            #######################
            self.frames_info[video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
            self.img_ids.extend([(video_name, frame_index) for frame_index in range(len(frames))])
        # import ipdb; ipdb.set_trace()
        logger.debug('data loader init: done')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        video_name, frame_index = img_ids_i

        img = []
        img_paths = []
        vid_len = len(self.frames_info[video_name])
        center_frame_name = self.frames_info[video_name][frame_index]
        # import ipdb; ipdb.set_trace()
        if self.perturb == 'shuffle':
            # This is added for temporal order analysis
            fl = [(x + vid_len) % vid_len for x in
                  range(frame_index - math.floor(float(self.num_frames) / 2), frame_index, 1)]
            fr = [(x + vid_len) % vid_len for x in
                  range(frame_index + 1, frame_index + math.ceil(float(self.num_frames) / 2), 1)]
            flr = fl + fr
            random.shuffle(flr)
            frame_indices = flr[:len(fl)] + [frame_index] + flr[len(fl):]
        elif self.perturb == 'repeat':
            frame_indices = [frame_index for _ in range(self.num_frames)]
        elif self.perturb == 'repeat_select':
            raise NotImplementedError
        elif self.perturb == 'random':
            frame_indices = random.choices(list(range(frame_index)) + list(range(frame_index + 1, vid_len)),
                                           k=self.num_frames)
            frame_indices[len(frame_indices) // 2] = frame_index

        else:
            frame_indices = [(x + vid_len) % vid_len for x in
                             range(frame_index - math.floor(float(self.num_frames) / 2),
                                   frame_index + math.ceil(float(self.num_frames) / 2), 1)]

        assert len(frame_indices) == self.num_frames
        frame_ids = []
        flows = []
        # import ipdb;ipdb.set_trace()
        for frame_id in frame_indices:
            frame_name = self.frames_info[video_name][frame_id]
            frame_ids.append(frame_name)
            img_path = os.path.join(self.dataset_path, video_name, frame_name + '.jpg')
            img_i = Image.open(img_path).convert('RGB')
            # img_i = img_i.resize((self.min_size,self.min_size))
            img.append(img_i)
            img_paths.append(img_path)
            ###################################
            if self.use_flow:
                """
                prev_frame = '%05d' % (int(frame_name) - 1)
                next_frame = '%05d' % (int(frame_name) + 1)
                fwd_flow_file = os.path.join(self.moca_flow_path, video_name,prev_frame + '_' + frame_name + '.png')
                bkd_flow_file = os.path.join(self.moca_flow_path, video_name, frame_name + '_' + next_frame + '.png')
                import ipdb;ipdb.set_trace()
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
                    raise Exception('Flow file not found for :%s-%s' % (video_name, frame_name))
                """
                flow_file = os.path.join(self.moca_flow_path, video_name, frame_name + '.png')
                prev_frame = '%05d' % (int(frame_name) - 1)
                next_frame = '%05d' % (int(frame_name) + 1)
                prev_flow_file = os.path.join(self.moca_flow_path, video_name, prev_frame + '.png')
                next_flow_file = os.path.join(self.moca_flow_path, video_name, next_frame + '.png')
                # import ipdb;ipdb.set_trace()
                if os.path.exists(flow_file):
                    flow = Image.open(flow_file).convert('RGB')
                elif os.path.exists(prev_flow_file):
                    flow = Image.open(prev_flow_file).convert('RGB')
                elif os.path.exists(next_flow_file):
                    flow = Image.open(next_flow_file).convert('RGB')
                else:
                    raise Exception('Flow file not found for :%s-%s' % (video_name, frame_name))
                flow = flow.resize((self.min_size, self.min_size))
                flows.append(flow)

        target = {'video_name': video_name, 'center_frame': center_frame_name, 'frame_ids': frame_ids,
                  'vid_len': vid_len, 'img_paths': img_paths}
        if self.use_flow:
            target['flows'] = flows
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.use_flow:
            target['flows'] = [target['flows'][i].unsqueeze(0) for i in range(len(img))]
            target['flows'] = torch.cat(target['flows'], dim=0)
        # import ipdb;ipdb.set_trace()
        return torch.cat(img, dim=0), target


class MoCAStaticDynamicPair(torch.utils.data.Dataset):

    def __init__(self, num_frames=6, min_size=473, sequence_names=None, perturb='repeat', n_factors=2):
        super(MoCAStaticDynamicPair, self).__init__()
        logger.debug('MoCAStaticDynamicPair----> perturb:%s' % perturb)
        # import ipdb;ipdb.set_trace()
        self.num_frames = num_frames
        self.min_size = min_size
        self.perturb = perturb  # used for interpretability experiments, use 'none' for other cases
        self.dataset_path = dataset_path_config.moca_dataset_images_path
        self.moca_flow_path = dataset_path_config.moca_dataset_flow_path
        self.files_list = dataset_path_config.moca_val_set_file
        self._transforms_norm = self.__class__.make_norm_transforms()
        self._transforms_noisy = self.__class__.make_noisy_transforms()
        self.n_factors = n_factors
        self.current_factor = 0
        logger.debug('self.perturb:%s' % self.perturb)
        logger.debug('self.n_factors:%d' % self.n_factors)

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
        for video_name in video_names:
            frames = sorted(glob.glob(os.path.join(self.dataset_path, video_name, '*.jpg')))
            # frames = frames[::2]
            # step=1
            # if len(frames)//self.num_frames> 1:
            #    # import ipdb;ipdb.set_trace()
            #    step = len(frames)//self.num_frames
            #    # frames = frames[::step] 
            self.frames_info[video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
            self.img_ids.extend([(video_name, frame_index) for frame_index in
                                 range(self.num_frames // 2, len(frames) - self.num_frames // 2, 1)])
        # import ipdb; ipdb.set_trace()
        logger.debug('data loader init: done')

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def make_norm_transforms(min_size=360):
        return T.Compose([
            T.RandomResize([min_size], max_size=int(1.8 * min_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def make_noisy_transforms(min_size=360):
        return T.Compose([
            T.RandomSizeCrop(min_size, max_size=int(1.8 * min_size)),
            # PerturbPhotometricDistort(),
            T.RandomResize([min_size], max_size=int(1.8 * min_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __load_frames__(self, p_video_name, p_frame_indices, noisy=False):
        # import ipdb;ipdb.set_trace()
        frame_indices = []
        frame_paths = []
        frame_list = []
        for frame_id in p_frame_indices:
            frame_name = self.frames_info[p_video_name][frame_id]
            frame_indices.append(frame_name)
            img_path = os.path.join(self.dataset_path, p_video_name, frame_name + '.jpg')
            img_i = Image.open(img_path).convert('RGB')
            # img_i = img_i.resize((self.min_size,self.min_size))
            frame_list.append(img_i)
            frame_paths.append(img_path)
        target = {'video_name': p_video_name, 'frame_ids': frame_indices}

        # import ipdb;ipdb.set_trace()
        if noisy:
            vid_clip, _ = self._transforms_noisy(frame_list, target)
            vid_clip = torch.stack(vid_clip, dim=0).permute(1, 0, 2, 3)
            # import ipdb;ipdb.set_trace()
            # std = vid_clip.std()
            # mean = vid_clip.mean()
            # noise = torch.tensor(np.random.normal(mean, std, vid_clip.size()), dtype=torch.float)
            # vid_clip = 0.95 * vid_clip + 0.05 * noise
        else:
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
        # logger.debug('factor:%d'%factor)
        meta_data = {
            'video_name': video_name,
            'center_frame_name': center_frame_name,
        }
        frame_indices = [(x + vid_len) % vid_len for x in
                         range(frame_index - math.floor(float(self.num_frames) / 2),
                               frame_index + math.ceil(float(self.num_frames) / 2), 1)]
        assert len(frame_indices) == self.num_frames
        vid_clip_1 = self.__load_frames__(video_name, frame_indices, noisy=False)
        # import ipdb;ipdb.set_trace()
        meta_data['frame_indices_1'] = [str(self.frames_info[video_name][indx]) for indx in frame_indices]
        if factor == 0:
            # Same motion different appearance
            # vid_clip_2 = copy.deepcopy(vid_clip_1)
            # meta_data['frame_indices_2'] = meta_data['frame_indices_2'] = [str(self.frames_info[video_name][indx]) for indx in frame_indices]

            frame_indices_2 = [(x + 0) % vid_len for x in frame_indices]
            meta_data['frame_indices_2'] = [str(self.frames_info[video_name][indx]) for indx in frame_indices_2]
            vid_clip_2 = self.__load_frames__(video_name, frame_indices_2, noisy=True)

            # vid_clip_2 = self.__load_frames__(video_name, frame_indices, noisy=True)
            # meta_data['frame_indices_2'] = [str(self.frames_info[video_name][indx]) for indx in frame_indices]
        elif factor == 1:
            # same appearance different motion
            ###################################
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
                frame_indices_2 = [frame_index for _ in range(self.num_frames)]
            elif self.perturb == 'random':
                frame_indices_2 = random.choices(list(range(frame_index)) + list(range(frame_index + 1, vid_len)),
                                                 k=self.num_frames)
                frame_indices_2[len(frame_indices_2) // 2] = frame_index
            else:
                raise NotImplementedError
            assert len(frame_indices_2) == self.num_frames
            meta_data['frame_indices_2'] = [str(self.frames_info[video_name][indx]) for indx in frame_indices_2]
            vid_clip_2 = self.__load_frames__(video_name, frame_indices_2, noisy=False)
        elif factor == 2:
            # Same motion, different appearance
            ###################################
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
                frame_indices_2 = [frame_index for _ in range(self.num_frames)]
            elif self.perturb == 'random':
                frame_indices_2 = random.choices(list(range(frame_index)) + list(range(frame_index + 1, vid_len)),
                                                 k=self.num_frames)
                frame_indices_2[len(frame_indices_2) // 2] = frame_index
            else:
                raise NotImplementedError
            assert len(frame_indices_2) == self.num_frames
            vid_clip_2 = self.__load_frames__(video_name, frame_indices_2, noisy=False)
            meta_data['frame_indices_2'] = meta_data['frame_indices_2'] = [str(self.frames_info[video_name][indx]) for
                                                                           indx in frame_indices_2]
        else:
            vid_clip_2 = copy.deepcopy(vid_clip_1)
            # import ipdb;ipdb.set_trace()
            meta_data['frame_indices_2'] = meta_data['frame_indices_2'] = [str(self.frames_info[video_name][indx]) for
                                                                           indx in frame_indices]
        # import ipdb; ipdb.set_trace()
        return factor, vid_clip_1, vid_clip_2, meta_data


def make_validation_transforms(min_size=360):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([min_size], max_size=int(1.8 * min_size)),
        normalize,
    ])


class PerturbPhotometricDistort(object):
    def __init__(self):
        self.pd = [
            T.ConvertColor(current='BGR', transform='HSV'),
            Saturation(),
            Hue(),
            T.ConvertColor(current='HSV', transform='BGR'),
        ]
        self.perms = ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, clip, target):
        imgs = []
        # distort = T.Compose(self.pd)
        # import ipdb;ipdb.set_trace()
        img_shape = np.asarray(clip[0]).shape
        # delta = rand.uniform(-self.delta, self.delta)

        delta = np.random.normal(0, 0.01, img_shape).astype('float32')
        alpha = rand.uniform(0.9, 1.1, img_shape[2:]).astype('float32')
        alpha = alpha[np.newaxis, np.newaxis, :]
        swap = self.perms[rand.randint(len(self.perms))]

        for img in clip:
            # delta = 2 * rand.normal(0, 1.0, img_shape).astype('float32')
            # alpha = rand.uniform(0.8, 1.2, img_shape[2:]).astype('float32')
            # alpha = alpha[np.newaxis, np.newaxis, :]
            # swap = self.perms[rand.randint(len(self.perms))]
            # import ipdb; ipdb.set_trace()
            img = np.asarray(img).astype('float32')
            img = (img + delta) * alpha  # + delta
            img[img > 255] = 255
            img[img <= 0] = 0
            # img, target = distort(img, target)
            img = img[:, :, swap] * 0.1 + img * 0.9
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, target


class Saturation(object):
    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, target


class Hue(object):  #
    def __init__(self, delta=5.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if True:
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target
