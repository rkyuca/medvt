# from ast import main
import os
import os.path as osp
import csv
import random
import mat73
import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from actor_action.datasets import path_config as dataset_path_config
from actor_action.datasets import transforms as T


def make_train_transform(train_size=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    toPIL = torch.data.transforms.toPIL()
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([500, 600, 700]),
            T.RandomSizeCrop(473, 750),
            T.RandomResize([train_size], max_size=int(1.8 * train_size)),  # for r50
        ]),
        normalize,
    ])


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)


def read_csv(input, flag=1):
    """
    Args:
        input: csv file
        flag : 1 indicates traing; 0 indicates testing
    """

    file_handler = open(input)
    csv_file = csv.reader(file_handler)
    rst = []
    for row in csv_file:
        if int(row[-1]) != flag:
            rst.append(
                [row[0], row[1], row[4], row[5], row[6], row[7]]
            )
    # return list of list: [id, label, height, width, num_frames, num_gt]
    # return [[row[0], row[1], row[4], row[5], row[6], row[7]] for row in csv_file if int(row[-1]) == flag]
    return rst


def read_mask(file_name):
    anot = mat73.loadmat(file_name)
    anot_parsed = anot['reS_id']
    return anot_parsed


def make_transform(size=224):
    T = torch.nn.Sequential(
        # transforms.Resize(size),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    # T = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    return T


def make_gt_transform(size=224):
    return transforms.Resize(size)


class A2dDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, frame_path, gt_path, context_len=3, train=True):
        super().__init__()
        """
        Args:
            csv_path: 
            frame_path:
            gt_path:
            Context_len: Half of the video frames input length, e.g., 8.
        """

        self.frame_path = frame_path
        self.gt_path = gt_path
        self.context_length = context_len
        self.data = read_csv(csv_path, flag=train)
        self.train = train

        self.transform = make_transform()
        self.gt_transform = make_gt_transform()
        assert osp.exists(self.frame_path)
        assert osp.exists(self.gt_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Obtain information        
        vid_id, vid_cls, _, _, vid_frames_cnt, _ = self.data[idx]
        vid_frames_cnt = int(vid_frames_cnt)

        # retrieve ground-truth, random select one index and load mat file
        gt_files = os.listdir(
            osp.join(self.gt_path, vid_id)
        )
        gt_file = random.choice(gt_files)

        # Load mat ground-truth file
        gt_img = read_mask(
            osp.join(
                self.gt_path, vid_id, gt_file
            )
        )

        # Select frames centered aroud the ground-truth id
        gt_id = int(gt_file.split('.')[0])

        # Judge is the selected gt_id is too small or too large
        if gt_id - self.context_length < 1:
            start_id, end_id = gt_id, gt_id + self.context_length * 2
            gt_index = 0
        elif gt_id + self.context_length > vid_frames_cnt:
            start_id, end_id = gt_id - self.context_length, gt_id
            gt_index = -1
        else:
            start_id, end_id = gt_id - self.context_length, gt_id + self.context_length # So the 3rd frame is the center frame!!!
            gt_index = 3

        # Load img sequences
        img_tensors = torch.stack([torchvision.io.read_image(
            osp.join(
                self.frame_path, vid_id, '{:05d}.png'.format(idx))) for idx in range(start_id, end_id)
        ], 0) / 255.0

        # run the transform
        img_tensors = self.transform(img_tensors.float())

        # Transform image to fixed size for now
        img_tensors = interpolate(img_tensors.float(), size=(320, 480), mode='bilinear')
        if self.train:
            gt_tensor = interpolate(torch.from_numpy(gt_img).float().unsqueeze(0).unsqueeze(0), size=(320, 480))
        else:
            gt_tensor = torch.from_numpy(gt_img).float().unsqueeze(0).unsqueeze(0)

        # Return
        return img_tensors.view(-1, 320, 480), torch.tensor(int(vid_cls), dtype=torch.int8), gt_tensor, vid_id, gt_index

class DebugA2dDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path) -> None:
        super().__init__()
        self.data = read_csv(csv_path)

    def forward(self):
        pass

if __name__ == '__main__':
    dataset = A2dDataset(
        csv_path='/local/riemann1/home/msiam/a2d_dataset/Release/videoset.csv',
        frame_path='/local/riemann1/home/msiam/a2d_dataset/frames',
        gt_path='/local/riemann1/home/msiam/a2d_dataset/Release/Annotations/mat'
    )
