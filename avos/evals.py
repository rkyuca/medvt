import logging
import collections
import time
import numpy as np
import cv2
import ast
import operator
import csv
import pandas as pd
from tqdm import tqdm
import os
import glob
import torch
from torch.utils.data import DataLoader

import avos.utils as utils
import avos.utils.misc
from avos.utils.misc import NestedTensor
from avos.datasets.test.davis16_val_data import Davis16ValDataset as Davis16ValDataset
from avos.datasets.test.moca import MoCADataset
from avos.datasets.test.youtube_objects import YouTubeObjects
from avos.datasets import path_config as dataset_path_config
from avos.visual.overlay import create_overlay

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_eval_save_dir_name_from_args(_args):
    _dir_name = 'infer_%s%3d%slpp_mode%d_sc%0.2f_%d' % (
        _args.dataset,
        _args.val_size,
        'msc' if _args.msc else 'ssc',
        _args.lprop_mode,
        _args.lprop_scale,
        int(time.time()))
    return _dir_name


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_predictions_flip_ms(model, samples, gt_shape, flows=None, ms=True, ms_gather='mean', flip=True,
                                flip_gather='mean', scales=None, sigmoid=True):
    # import ipdb;ipdb.set_trace()
    outputs = compute_predictions_ms(model, samples, flows, gt_shape, ms=ms, ms_gather=ms_gather,
                                     scales=scales, sigmoid=sigmoid)
    outputs['pred_masks'] = utils.misc.interpolate(outputs['pred_masks'], size=gt_shape, mode="bilinear",
                                                   align_corners=False)
    if flip:
        # print('using flip')
        samples_flipped, flows_flipped = augment_flip(samples, flows)

        outputs_flipped = compute_predictions_ms(model, samples_flipped, flows_flipped, gt_shape, ms=ms,
                                                 ms_gather=ms_gather, scales=scales)
        outputs_flipped['pred_masks'] = utils.misc.interpolate(outputs_flipped['pred_masks'], size=gt_shape,
                                                               mode="bilinear", align_corners=False)
        if flip_gather == 'max':
            outputs['pred_masks'] = torch.max(outputs_flipped['pred_masks'].flip(-1), outputs['pred_masks'])
        else:
            outputs['pred_masks'] = (outputs_flipped['pred_masks'].flip(-1) + outputs['pred_masks']) / 2.0
    return outputs


def compute_predictions_ms(model, samples, flows, gt_shape, ms=True, ms_gather='mean',
                           scales=None, sigmoid=True):
    if scales is None:
        scales = [1]
    mask_list = []
    org_shape = samples.tensors.shape[-2:]
    # import ipdb;ipdb.set_trace()
    for scale in scales:
        size = [int(val * scale) for val in org_shape]
        tensors = samples.tensors
        mask = samples.mask
        flow_tensor = None
        if scale != 1:
            tensors = utils.misc.interpolate(tensors, size=size, mode="bilinear", align_corners=False)
            mask = utils.misc.interpolate(mask.unsqueeze(1).long().float(), size=size, mode="nearest").squeeze(1)
            mask[mask > 0.5] = True
            mask[mask <= 0.5] = False
            mask = mask.bool()
        if flows is not None:
            flow_tensor = utils.misc.interpolate(flows, size=size, mode="bilinear", align_corners=False)
        ms_sample = utils.misc.NestedTensor(tensors, mask)
        with torch.no_grad():
            model_output = model(ms_sample, flows=flow_tensor)
        pred = utils.misc.interpolate(model_output['pred_masks'], size=gt_shape, mode="bilinear", align_corners=False)
        if sigmoid:
            pred = pred.sigmoid()
        mask_list.append(pred)
    if ms:
        if ms_gather == 'max':
            ms_pred = torch.max(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
        else:
            ms_pred = torch.mean(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
    else:
        output_result = {'pred_masks': mask_list[0]}
    return output_result

def augment_flip(samples, flows, dim=-1):
    tensors = samples.tensors.clone().detach().flip(dim)
    mask = samples.mask.clone().detach().flip(dim)
    samples_flipped = NestedTensor(tensors, mask)
    flows_flipped = None
    if flows is not None:
        flows_flipped = flows.clone().detach().flip(dim)
    return samples_flipped, flows_flipped


def moca_read_annotation(annotation):
    reader = csv.reader(open(annotation, 'r'))
    next(reader, None)
    d = {}
    reader = sorted(reader, key=operator.itemgetter(1))
    for row in reader:
        _, fn, bbox, motion = row
        if bbox != '[]':
            if motion == '{}':
                motion = old_motion
            old_motion = motion
            name = fn.split('/')[-2]
            number = fn.split('/')[-1][:-4]
            if name not in d:
                d[name] = {}
            if number not in d[name]:
                d[name][number] = {}
            d[name][number]['fn'] = fn
            motion = ast.literal_eval(motion)
            d[name][number]['motion'] = motion['1']
            bbox = ast.literal_eval(bbox)
            _, xmin, ymin, width, height = list(bbox)
            xmin = max(xmin, 0.)
            ymin = max(ymin, 0.)
            d[name][number]['xmin'] = xmin
            d[name][number]['xmax'] = xmin + width
            d[name][number]['ymin'] = ymin
            d[name][number]['ymax'] = ymin + height
    return d


def moca_bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def moca_heuristic_fg_bg(mask):
    mask = mask.copy()
    h, w = mask.shape
    mask[1:-1, 1:-1] = 0
    borders = 2 * h + 2 * w - 4
    return np.sum(mask > 127.5) / borders


def moca_eval(out_dir='./results/moca', resize=1):
    moca_dataset_path = dataset_path_config.moca_dataset_images_path
    moca_csv = dataset_path_config.moca_annotations_csv
    moca_pred_dir = os.path.join(out_dir)
    if not os.path.exists(moca_pred_dir):
        raise ValueError('MoCA Pred Dir not exists!!!')

    out_csv = os.path.join(out_dir, 'MoCA_results.csv')

    with open(out_csv, 'w') as f:
        df = pd.DataFrame([], columns=['Seq_name', 'Locomotion_IoU', 'Locomotion_S_0.5', 'Locomotion_S_0.6',
                                       'Locomotion_S_0.7', 'Locomotion_S_0.8', 'Locomotion_S_0.9',
                                       'Deformation_IoU', 'Deformation_S_0.5', 'Deformation_S_0.6', 'Deformation_S_0.7',
                                       'Deformation_S_0.8', 'Deformation_S_0.9',
                                       'Static_IoU', 'Static_S_0.5', 'Static_S_0.6', 'Static_S_0.7', 'Static_S_0.8',
                                       'Static_S_0.9',
                                       'All_motion_IoU', 'All_motion_S_0.5', 'All_motion_S_0.6', 'All_motion_S_0.7',
                                       'All_motion_S_0.8', 'All_motion_S_0.9',
                                       'locomotion_num', 'deformation_num', 'static_num'])

        df.to_csv(f, index=False,
                  columns=['Seq_name', 'Locomotion_IoU', 'Locomotion_S_0.5', 'Locomotion_S_0.6', 'Locomotion_S_0.7',
                           'Locomotion_S_0.8', 'Locomotion_S_0.9',
                           'Deformation_IoU', 'Deformation_S_0.5', 'Deformation_S_0.6', 'Deformation_S_0.7',
                           'Deformation_S_0.8', 'Deformation_S_0.9',
                           'Static_IoU', 'Static_S_0.5', 'Static_S_0.6', 'Static_S_0.7', 'Static_S_0.8', 'Static_S_0.9',
                           'All_motion_IoU', 'All_motion_S_0.5', 'All_motion_S_0.6', 'All_motion_S_0.7',
                           'All_motion_S_0.8', 'All_motion_S_0.9',
                           'locomotion_num', 'deformation_num', 'static_num'])
        pass

    annotations = moca_read_annotation(moca_csv)
    Js = AverageMeter()

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    success_rates_overall = np.zeros(5)
    total_frames_l = 0
    total_frames_d = 0
    total_frames_s = 0
    success_l_overall = [0, 0, 0, 0, 0]
    success_d_overall = [0, 0, 0, 0, 0]
    success_s_overall = [0, 0, 0, 0, 0]

    video_names = sorted(os.listdir(moca_pred_dir))
    vid_iou_dict = {}
    for video in video_names:
        if video not in annotations:
            continue
        res_path = os.path.join(moca_pred_dir, video)
        res_list = sorted([f for f in glob.glob(res_path + '/*.png', recursive=False)])  # for our model

        n_frames = len(res_list)
        js = []

        iou_static = AverageMeter()
        iou_locomotion = AverageMeter()
        iou_deformation = AverageMeter()
        ns = 0
        nl = 0
        nd = 0
        success_l = [0, 0, 0, 0, 0]
        success_d = [0, 0, 0, 0, 0]
        success_s = [0, 0, 0, 0, 0]
        if resize:
            image = np.array(cv2.imread(os.path.join(moca_dataset_path, video, '{:05d}.jpg'.format(0))))
            H, W, _ = image.shape
        for ff in range(n_frames):
            # import ipdb;ipdb.set_trace()
            number = res_list[ff].split('/')[-1].split('.')[0]
            # number = str(ff).zfill(5)
            if number in annotations[video]:
                # get annotation
                motion = annotations[video][number]['motion']
                x_min = annotations[video][number]['xmin']
                x_max = annotations[video][number]['xmax']
                y_min = annotations[video][number]['ymin']
                y_max = annotations[video][number]['ymax']
                bbox_gt = [x_min, y_min, x_max, y_max]

                # get mask
                mask = np.array(cv2.imread(res_list[ff]), dtype=np.uint8)
                if len(mask.shape) > 2:
                    mask = mask.mean(2)
                H_, W_ = mask.shape

                if moca_heuristic_fg_bg(mask) > 0.5:
                    mask = (255 - mask).astype(np.uint8)

                thres = 0.1 * 255
                mask[mask > thres] = 255
                mask[mask <= thres] = 0

                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                area = 0

                for cnt in contours:
                    (x_, y_, w_, h_) = cv2.boundingRect(cnt)
                    area_ = np.sum(mask[y_:y_ + h_, x_:x_ + w_])
                    if area_ > area:
                        x = x_
                        y = y_
                        w = w_
                        h = h_
                        area = area_
                H_, W_ = mask.shape
                if area > 0:
                    bbox = np.array([x, y, x + w, y + h], dtype=float)
                    # if the size reference for the annotation (the original jpg image) is different from the size of the mask
                    if resize:
                        bbox[0] *= W / W_
                        bbox[2] *= W / W_
                        bbox[1] *= H / H_
                        bbox[3] *= H / H_
                    iou = moca_bb_intersection_over_union(np.array(bbox_gt, dtype=float), np.array(bbox, dtype=float))
                else:
                    iou = 0.
                js.append(iou)
                ################################################
                if video not in vid_iou_dict:
                    vid_iou_dict[video] = collections.OrderedDict()
                # import ipdb;ipdb.set_trace()
                vid_iou_dict[video][ff] = max(vid_iou_dict[video].get(ff, 0), iou)

                ####################################################

                # get motion
                if motion == '1':
                    iou_deformation.update(iou)
                    nd += 1
                    for k in range(len(thresholds)):
                        success_d[k] += int(iou > thresholds[k])

                elif motion == '0':
                    iou_locomotion.update(iou)
                    nl += 1
                    for k in range(len(thresholds)):
                        success_l[k] += int(iou > thresholds[k])

                elif motion == '2':
                    iou_static.update(iou)
                    ns += 1
                    for k in range(len(thresholds)):
                        success_s[k] += int(iou > thresholds[k])

        total_frames_l += nl
        total_frames_s += ns
        total_frames_d += nd
        for k in range(len(thresholds)):
            success_l_overall[k] += success_l[k]
            success_s_overall[k] += success_s[k]
            success_d_overall[k] += success_d[k]

        js_m = np.average(js)
        locomotion_val = -1.
        static_val = -1.
        deformation_val = -1.
        if iou_locomotion.count > 0:
            locomotion_val = iou_locomotion.avg
        if iou_deformation.count > 0:
            deformation_val = iou_deformation.avg
        if iou_static.count > 0:
            static_val = iou_static.avg

        all_motion_S = np.array(success_l) + np.array(success_s) + np.array(success_d)
        success_rates_overall += all_motion_S
        with open(out_csv, 'a') as f:
            raw_data = {'Seq_name': video, 'Locomotion_IoU': [locomotion_val],
                        'Locomotion_S_0.5': [success_l[0]], 'Locomotion_S_0.6': [success_l[1]],
                        'Locomotion_S_0.7': [success_l[2]], 'Locomotion_S_0.8': [success_l[3]],
                        'Locomotion_S_0.9': [success_l[4]],
                        'Deformation_IoU': [deformation_val],
                        'Deformation_S_0.5': [success_d[0]], 'Deformation_S_0.6': [success_d[1]],
                        'Deformation_S_0.7': [success_d[2]], 'Deformation_S_0.8': [success_d[3]],
                        'Deformation_S_0.9': [success_d[4]],
                        'Static_IoU': [static_val],
                        'Static_S_0.5': [success_s[0]], 'Static_S_0.6': [success_s[1]],
                        'Static_S_0.7': [success_s[2]], 'Static_S_0.8': [success_s[3]],
                        'Static_S_0.9': [success_s[4]],
                        'All_motion_IoU': [js_m],
                        'All_motion_S_0.5': [all_motion_S[0]], 'All_motion_S_0.6': [all_motion_S[1]],
                        'All_motion_S_0.7': [all_motion_S[2]], 'All_motion_S_0.8': [all_motion_S[3]],
                        'All_motion_S_0.9': [all_motion_S[4]],
                        'locomotion_num': [nl], 'deformation_num': [nd], 'static_num': [ns]}
            df = pd.DataFrame(raw_data, columns=['Seq_name', 'Locomotion_IoU', 'Locomotion_S_0.5', 'Locomotion_S_0.6',
                                                 'Locomotion_S_0.7', 'Locomotion_S_0.8', 'Locomotion_S_0.9',
                                                 'Deformation_IoU', 'Deformation_S_0.5', 'Deformation_S_0.6',
                                                 'Deformation_S_0.7', 'Deformation_S_0.8', 'Deformation_S_0.9',
                                                 'Static_IoU', 'Static_S_0.5', 'Static_S_0.6', 'Static_S_0.7',
                                                 'Static_S_0.8', 'Static_S_0.9',
                                                 'All_motion_IoU', 'All_motion_S_0.5', 'All_motion_S_0.6',
                                                 'All_motion_S_0.7', 'All_motion_S_0.8', 'All_motion_S_0.9',
                                                 'locomotion_num', 'deformation_num', 'static_num'])
            df.to_csv(f, header=False, index=False)
        Js.update(js_m)

    ##########################################
    # import ipdb;ipdb.set_trace()
    """
    import csv
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    frame_iou_file = os.path.join(out_dir, 'moca_frame_wise_iou.csv')
    logger.debug('frame_iou_file:%s'%frame_iou_file)
    # collections.OrderedDict()
    with open(frame_iou_file, 'w') as f:
        cf = csv.writer(f)
        for kk,vv in vid_iou_dict.items():
            # row = [kk]+list(vv.values())
            row = list(vv.values())
            cf.writerow(row)
            mean_iou = np.mean(list(vv.values()))
            logger.debug('video: %s  iou:%0.3f'%(kk, mean_iou))
            # cf.writerow(video_ious)
    """
    ###########################################
    success_rates_overall = success_rates_overall / (total_frames_l + total_frames_s + total_frames_d)
    info = 'Overall :  Js: ({:.3f}). SR at '
    for k in range(len(thresholds)):
        info += str(thresholds[k])
        info += ': ({:.3f}), '
    info = info.format(Js.avg, success_rates_overall[0], success_rates_overall[1], success_rates_overall[2],
                       success_rates_overall[3], success_rates_overall[4])
    logger.debug('dataset: MoCA result:%s' % info)
    logger.debug('dataset: MoCA mean_iou:%0.3f' % Js.avg)
    return Js.avg, info


def moca_infer(model, data_loader, device, msc=False, flip=False, save_pred=False, out_dir='./results/moca/',
               videos=None):
    # import ipdb;ipdb.set_trace()
    # msc = False  # #TODO check, msc did not improve here.
    if msc:
        # _scales = [0.7, 0.8, 0.9, 1, 1.1, 1.2]
        _scales = [0.9, 0.95, 1.0, 1.05, 1.1]
    else:
        _scales = [1]
    logger.debug('using scales: %s'%str(_scales))
    model.eval()
    i_iter = 0
    moca_csv = dataset_path_config.moca_annotations_csv
    annotations = moca_read_annotation(moca_csv)
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_name = targets[0]['video_name']
        center_frame = targets[0]['center_frame']
        if videos is not None and video_name not in videos:
            continue
        if video_name not in annotations:
            continue
        if center_frame not in annotations[video_name]:
            continue
        frame_ids = targets[0]['frame_ids']
        center_frame_index = frame_ids.index(center_frame)
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        gt_shape = samples.tensors.shape[-2:]
        outputs = compute_predictions_flip_ms(model, samples, gt_shape, flows=None, ms=msc, ms_gather='mean',
                                              flip=flip, flip_gather='max',
                                              scales=_scales)
        src_masks = outputs["pred_masks"]
        yc = src_masks[0].cpu().detach().numpy().copy()
        mask = yc[center_frame_index, :, :]
        mask_th = 0.5  # same threshold as davis 
        mask[mask > mask_th] = 255
        mask[mask <= mask_th] = 0
        mask = mask.astype(np.uint8)
        if save_pred:
            pred_out_dir = os.path.join(out_dir, video_name)
            if not os.path.exists(pred_out_dir):
                os.makedirs(pred_out_dir)
            cv2.imwrite(os.path.join(pred_out_dir, '%s.png' % center_frame), mask)
    return


@torch.no_grad()
def infer_ytbobj_perseqpercls(model, data_loader, device, msc=False, flip=False, save_pred=False, save_gt_overlay=False,
                              out_dir='./results/youtube_objects/'):
    if msc:
        _scales = [0.95, 1, 1.05, 1.1, 1.15]  # ResNet 75.2 # Swin 79.1
    else:
        _scales = [1]
    # logger.debug('_scales: '+str(_scales))
    model.eval()
    i_iter = 0
    percls_perseq_iou_dict = {}
    num_iou_dict = {}
    total_mask = 0
    running_video_name = None
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_name = targets[0]['video_name']
        video_name = video_name.split('/')[1]
        frame_ids = targets[0]['frame_ids']
        center_frame_name = targets[0]['center_frame']
        center_frame_index = frame_ids.index(center_frame_name)
        center_img_path = targets[0]['img_paths'][center_frame_index]
        img_dir, frame_name = os.path.split(center_img_path)
        mask_dir = img_dir.replace('frames', 'youtube_masks')
        frame_no = frame_name.replace('frame', '').replace('.jpg', '')
        mask_frame_name = '%05d.jpg' % int(frame_no)
        mask_file = os.path.join(mask_dir, 'labels', mask_frame_name)
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        if running_video_name is not None and video_name != running_video_name:
            cls_mean = np.mean([seq_sum / count for seq_sum, count in
                                zip(percls_perseq_iou_dict[running_video_name].values(),
                                    num_iou_dict[running_video_name].values())])
            logger.debug('class_name:%s  iou:%0.3f' % (running_video_name, cls_mean))
        running_video_name = video_name
        tokens2 = mask_file.split('/')
        object_ = tokens2[-7]
        seq_name = tokens2[-5]
        if object_ not in percls_perseq_iou_dict:
            percls_perseq_iou_dict[object_] = {}
            num_iou_dict[object_] = {}
        if seq_name not in percls_perseq_iou_dict[object_]:
            percls_perseq_iou_dict[object_][seq_name] = 0
            num_iou_dict[object_][seq_name] = 0
        if not os.path.exists(mask_file):
            continue
        total_mask += 1
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        gt_shape = mask.shape
        th_mask = 80
        mask[mask < th_mask] = 0
        mask[mask >= th_mask] = 1
        th_pred = 0.5  # 110.0 / 255.0  # 110 / 255.0
        torch.cuda.empty_cache()
        outputs = compute_predictions_flip_ms(model, samples, gt_shape, flows=None, ms=msc, ms_gather='mean',
                                              flip=flip, flip_gather='max', scales=_scales)
        torch.cuda.empty_cache()
        src_masks = outputs["pred_masks"]
        yc_logits = src_masks[0].cpu().detach().numpy()[center_frame_index, :, :].copy()
        bin_mask = yc_logits.copy()
        bin_mask[bin_mask < th_pred] = 0
        bin_mask[bin_mask >= th_pred] = 1
        out = bin_mask.astype(mask.dtype)
        # ###########################
        iou = eval_iou(mask.copy(), out.copy())
        percls_perseq_iou_dict[object_][seq_name] += iou
        num_iou_dict[object_][seq_name] += 1
        """
        if save_pred:
            pred_out_dir = os.path.join(out_dir, '/'.join(
                mask_dir.split('/')[-5:]))  # this one follows the same dir struck as gt
            if not os.path.exists(pred_out_dir):
                os.makedirs(pred_out_dir)
            cv2.imwrite(os.path.join(pred_out_dir, '%s.png' % center_frame_name),
                        out.astype(np.float32) * 255)  # it was 0, 1
        """
        if save_pred:
            # # Save logits ##########################
            logits_out_dir = os.path.join(out_dir,'logits', '/'.join(
                mask_dir.split('/')[-5:]))  # this one follows the same dir struck as gt
            if not os.path.exists(logits_out_dir):
                os.makedirs(logits_out_dir)
            cv2.imwrite(os.path.join(logits_out_dir, '%s.png' % center_frame_name),
                        (yc_logits.astype(np.float32) * 255).astype(np.uint8))
            # # Save binary masks ####################################
            bm_out_dir = os.path.join(out_dir,'bin_mask', '/'.join(
                mask_dir.split('/')[-5:]))  # this one follows the same dir struck as gt
            if not os.path.exists(bm_out_dir):
                os.makedirs(bm_out_dir)
            cv2.imwrite(os.path.join(bm_out_dir, '%s.png' % center_frame_name),
                        (out.astype(np.float32) * 255).astype(np.uint8))  # it is 0, 1
            # # Save overlay ###############################
            # import ipdb;ipdb.set_trace()
            overlay_mask = (out.copy().astype(np.float32)* 255).astype(np.uint8)
            if overlay_mask.max() > 1:
                overlay_mask[overlay_mask > 100] = 255
                overlay_mask[overlay_mask <= 100] = 0
            center_img_path = targets[0]['img_paths'][center_frame_index]
            center_img = cv2.imread(center_img_path)
            if overlay_mask.shape[0]!=center_img.shape[0] or overlay_mask.shape[1]!=center_img.shape[1]:
                center_img = cv2.resize(center_img, overlay_mask.shape[::-1])
            overlay = create_overlay(center_img, overlay_mask, [0, 255])
            overlay_out_dir = os.path.join(out_dir,'overlay', '/'.join(
                mask_dir.split('/')[-5:]))  # this one follows the same dir struck as gt
            if not os.path.exists(overlay_out_dir):
                os.makedirs(overlay_out_dir)
            cv2.imwrite(os.path.join(overlay_out_dir, '%s.png' % center_frame_name),overlay)

        if save_gt_overlay:
            # import ipdb;ipdb.set_trace()
            overlay_mask = (mask.copy().astype(np.float32)* 255).astype(np.uint8)
            if overlay_mask.max() > 1:
                overlay_mask[overlay_mask > 100] = 255
                overlay_mask[overlay_mask <= 100] = 0
            center_img_path = targets[0]['img_paths'][center_frame_index]
            center_img = cv2.imread(center_img_path)
            if overlay_mask.shape[0]!=center_img.shape[0] or overlay_mask.shape[1]!=center_img.shape[1]:
                center_img = cv2.resize(center_img, overlay_mask.shape[::-1])
            overlay = create_overlay(center_img, overlay_mask, [0, 255])
            overlay_out_dir = os.path.join(out_dir,'gt_overlay', '/'.join(
                mask_dir.split('/')[-5:]))  # this one follows the same dir struck as gt
            if not os.path.exists(overlay_out_dir):
                os.makedirs(overlay_out_dir)
            cv2.imwrite(os.path.join(overlay_out_dir, '%s.png' % center_frame_name),overlay)

    percls_perseq = np.mean([seq_sum / count for seq_sum, count in
                             zip(percls_perseq_iou_dict[running_video_name].values(),
                                 num_iou_dict[running_video_name].values())])
    logger.debug('class_name:%s iou:%0.3f' % (running_video_name, percls_perseq))
    logger.debug('total_masks:%d' % total_mask)
    # ### ### Write the results to CSV ### ###
    logger.debug('****************************************************************')
    logger.debug('***************Youtube-Objects Eval Results**********************')
    iou_objs = {}
    for obj in percls_perseq_iou_dict.keys():
        for seq in percls_perseq_iou_dict[obj].keys():
            percls_perseq_iou_dict[obj][seq] /= num_iou_dict[obj][seq]
        iou_objs[obj] = np.mean(list(percls_perseq_iou_dict[obj].values()))
    overall_iou = np.mean(list(iou_objs.values()))
    for obj, obj_iou in iou_objs.items():
        logger.debug('IoU reported for  %s is %0.3f' % (obj, obj_iou))
    logger.debug('Youtube Objects Average IoU : %0.3f' % overall_iou)
    logger.debug('****************************************************************')
    # write_youtubeobjects_results_to_csv(out_dir, 'youtube_objects_results.csv', iou_objs)
    # write results to csv
    csv_file_name = 'youtube_objects_results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    obj_names = []
    obj_iou = []
    for k, v in iou_objs.items():
        obj_names.append(k)
        obj_iou.append('%0.3f' % v)
    obj_names.append('Overall')
    obj_iou.append('%0.3f' % np.mean(list(iou_objs.values())))
    with open(os.path.join(out_dir, csv_file_name), 'w') as f:
        cf = csv.writer(f)
        cf.writerow(obj_names)
        cf.writerow(obj_iou)
    return


def eval_iou(annotation, segmentation):
    """
    Collected from https://github.com/fperazzi/davis/blob/main/python/lib/davis/measures/jaccard.py
    Compute region similarity as the Jaccard Index.
         Arguments:
             annotation   (ndarray): binary annotation   map.
             segmentation (ndarray): binary segmentation map.
         Return:
             jaccard (float): region similarity
    """
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
            np.sum((annotation | segmentation), dtype=np.float32)


@torch.no_grad()
def infer_on_davis(model, data_loader, device, msc=False, flip=False, save_pred=False,
                   out_dir='./results/davis/', msc_scales=None):
    if msc and msc_scales is not None:
        _scales = msc_scales
    elif msc and msc_scales is None:
        _scales = [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
    else:
        _scales = [1]
    model.eval()
    i_iter = 0
    iou_list = []
    vid_iou_dict = {}
    running_video_name = None
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_name = targets[0]['video_name']
        frame_ids = targets[0]['frame_ids']
        center_frame_name = targets[0]['center_frame']
        center_frame_index = frame_ids.index(center_frame_name)
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks','flows'] else v for k, v in t.items()} for t in targets]
        # import ipdb;ipdb.set_trace()
        flows = None
        if 'flows' in targets[0]:
            flows = torch.stack([ t['flows'] for t in targets ]).squeeze(0)
            # print('received flow...')
        # ###############################################
        center_gt_path = targets[0]['mask_paths'][center_frame_index]
        center_gt = cv2.imread(center_gt_path, cv2.IMREAD_GRAYSCALE)
        center_gt[center_gt > 0] = 1
        gt_shape = center_gt.shape
        if running_video_name is not None and video_name != running_video_name:
            video_iou = np.mean(list(vid_iou_dict[running_video_name].values()))
            logger.debug('video_name:%s iou:%0.3f' % (running_video_name, video_iou))
        running_video_name = video_name
        outputs = compute_predictions_flip_ms(model, samples, gt_shape, flows=flows,  ms=msc, ms_gather='mean',
                                              flip=flip, flip_gather='mean', scales=_scales)
        # import ipdb; ipdb.set_trace()
        src_masks = outputs["pred_masks"]
        yc_logits = src_masks.squeeze(0).cpu().detach().numpy()[center_frame_index, :, :].copy()
        yc_binmask = yc_logits.copy()
        yc_binmask[yc_binmask > 0.5] = 1
        yc_binmask[yc_binmask <= 0.5] = 0
        out = yc_binmask.astype(center_gt.dtype)
        # ########################################
        iou = eval_iou(center_gt.copy(), out.copy())
        iou_list.append(iou)
        if video_name not in vid_iou_dict:
            vid_iou_dict[video_name] = {}
        vid_iou_dict[video_name][center_frame_name] = iou
        if save_pred:
            logits_out_dir = os.path.join(out_dir, 'logits', video_name)
            if not os.path.exists(logits_out_dir):
                os.makedirs(logits_out_dir)
            cv2.imwrite(os.path.join(logits_out_dir, '%s.png' % center_frame_name),
                        (yc_logits.astype(np.float32) * 255).astype(np.uint8))
            bm_out_dir = os.path.join(out_dir, 'bin_mask', video_name)
            if not os.path.exists(bm_out_dir):
                os.makedirs(bm_out_dir)
            cv2.imwrite(os.path.join(bm_out_dir, '%s.png' % center_frame_name),
                        (out.astype(np.float32) * 255).astype(np.uint8))  # it is 0, 1
            # ### save overlay
            overlay_mask = (out.copy().astype(np.float32)* 255).astype(np.uint8)
            if overlay_mask.max() > 1:
                overlay_mask[overlay_mask > 100] = 255
                overlay_mask[overlay_mask <= 100] = 0
            center_img_path = targets[0]['img_paths'][center_frame_index]
            center_img = cv2.imread(center_img_path)
            overlay = create_overlay(center_img, overlay_mask, [0, 255])
            overlay_out_dir = os.path.join(out_dir, 'overlay', video_name)
            if not os.path.exists(overlay_out_dir):
                os.makedirs(overlay_out_dir)
            cv2.imwrite(os.path.join(overlay_out_dir, '%s.png' % center_frame_name),overlay)

    video_iou = np.mean(list(vid_iou_dict[running_video_name].values()))
    logger.debug('video_name:%s iou:%0.3f' % (running_video_name, video_iou))
    video_mean_iou = np.mean([np.mean(list(vid_iou_f.values())) for _, vid_iou_f in vid_iou_dict.items()])
    # ### ### Write the results to CSV ### ###
    csv_file_name = 'davis_results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_names = []
    video_ious = []
    for k, v in vid_iou_dict.items():
        vid_iou = np.mean(list(v.values()))
        video_names.append(k)
        video_ious.append('%0.3f' % vid_iou)
        logger.debug('video_name:%s iou:%0.3f' % (k, vid_iou))
    video_names.append('Video Mean')
    video_ious.append('%0.3f' % video_mean_iou)
    with open(os.path.join(out_dir, csv_file_name), 'w') as f:
        cf = csv.writer(f)
        cf.writerow(video_names)
        cf.writerow(video_ious)
    logger.debug('Davis Videos Mean IOU: %0.3f' % video_mean_iou)
    return video_mean_iou


def run_inference(args, device, model, load_state_dict=True, out_dir=None, videos=None):
    if out_dir is None:
        out_dir = args.output_dir
    if out_dir is None or len(out_dir) == 0:
        out_dir = './results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # ### Data Loader #########
    if not hasattr(args, 'save_pred'):
        args.save_pred = False
    if not hasattr(args, 'msc'):
        args.msc = False
    if not hasattr(args, 'flip'):
        args.flip = False
    if not hasattr(args, 'style'):
        args.style = None
    if not hasattr(args, 'perturb'):
        args.perturb = 'none'
    if not hasattr(args, 'sampling_stride'):
        args.sampling_stride = 1
    if not hasattr(args,'save_gt_overlay'):
        args.save_gt_overlay=False

    if hasattr(args,'davis_input_max_sc') and args.msc:
        davis_max_sc = args.davis_input_max_sc
    else :
        davis_max_sc = None


    if args.dataset == 'davis':
        dataset_val = Davis16ValDataset(num_frames=args.num_frames, val_size=args.val_size,
                                        sequence_names=args.sequence_names,
                                        max_sc=davis_max_sc, style=args.style,
                                        use_flow=args.use_flow)
    elif args.dataset == 'moca':
        dataset_val = MoCADataset(num_frames=args.num_frames, min_size=args.val_size,
                                  sequence_names=args.sequence_names, perturb=args.perturb,
                                  sampling_stride=args.sampling_stride,
                                  use_flow=args.use_flow)
    elif args.dataset == 'ytbo':
        dataset_val = YouTubeObjects(num_frames=args.num_frames, min_size=args.val_size,
                                     use_flow=args.use_flow)
    else:
        raise ValueError('Dataset not implemented')
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=utils.misc.collate_fn,
                                 num_workers=args.num_workers)
    with torch.no_grad():
        if load_state_dict:
            state_dict = torch.load(args.model_path)['model']
            model.load_state_dict(state_dict, strict=True)
            logger.debug('model checkpoint loaded ...')
        model.eval()
        # import ipdb;ipdb.set_trace()
        if args.dataset == 'moca':
            moca_infer(model, data_loader_val, device, msc=args.msc, flip=args.flip, save_pred=True,
                       out_dir=out_dir, videos=videos)
            moca_eval(out_dir=out_dir, resize=1)
        elif args.dataset == 'ytbo':
            infer_ytbobj_perseqpercls(model, data_loader_val, device, msc=args.msc, flip=args.flip,
                                      save_pred=args.save_pred,save_gt_overlay=args.save_gt_overlay,
                                      out_dir=out_dir)
        elif args.dataset == 'davis':
            infer_on_davis(model, data_loader_val, device, msc=args.msc, flip=args.flip,
                           save_pred=args.save_pred, out_dir=out_dir,
                           msc_scales=args.davis_msc_scales if hasattr(args, 'davis_msc_scales') else None)
        else:
            raise ValueError('dataset name: %s not implemented' % args.dataset)


def inference_on_all_vos_dataset(args, device, model, datasets=None, val_sizes=None, _load_state=True):
    if datasets is None:
        vos_datasets = ['davis', 'ytbo', 'moca']
    else:
        vos_datasets = datasets
    args.sequence_names = None
    if val_sizes is None:
        val_sizes = {'davis': 473, 'ytbo': 360, 'moca': 473}
    base_output_dir = args.output_dir if args.output_dir is not None else './results'
    msc = hasattr(args, 'msc') and args.msc
    flip = hasattr(args, 'flip') and args.flip
    args.msc = msc
    args.flip = flip
    for data_set_name in vos_datasets:
        args.dataset = data_set_name
        args.val_size = val_sizes[data_set_name]
        out_dir_name = create_eval_save_dir_name_from_args(args)
        output_dir = os.path.join(base_output_dir, out_dir_name)
        logger.debug('##########################################################')
        # logger.debug(args)
        logger.debug('Doing inference on best checkpoint')
        logger.debug(f'Inference on {args.dataset} using val_size:{args.val_size} msc:{args.msc} flip:{args.flip}')
        logger.debug('****************************************************************')
        logger.debug(f'Inference on {args.dataset} using val_size:{args.val_size} msc:{args.msc} flip:{args.flip}')
        torch.cuda.empty_cache()
        run_inference(args, device, model, load_state_dict=_load_state, out_dir=output_dir)
        torch.cuda.empty_cache()
        logger.debug('****************************************************************')
