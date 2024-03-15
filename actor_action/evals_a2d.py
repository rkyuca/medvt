import sys
import numpy as np
import logging
import torch
import math
import colorsys
import random
import os
import cv2
import time
from PIL import Image
import torch.backends.cudnn as cudnn
import avos.utils.misc as utils
from actor_action.utils import metric
# from common import get_dataset_colormap

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cudnn.benchmark = False
cudnn.deterministic = True


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def PIL2array(img, dim):
    # dim is 3 for RGB, 4 for RGBA
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], dim)


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch: int, output_viz_dir=None,
             total_epochs=15, save_masks=False, n_classes=80, metadata=None, save_gt=False, verbose=False):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test Epoch: [{}/{}]:'.format(epoch, total_epochs)
    i_iter = 0
    running_video_name = None
    iou_dict = {}
    for samples, vid_cls, targets, vid_id, gt_index, target_im_path, gt_path in metric_logger.log_every(data_loader,
                                                                                                        500, header):
        i_iter = i_iter + 1
        video_name = vid_id[0]
        # import ipdb;ipdb.set_trace()
        target_frame_name = target_im_path[0].split('/')[-1].split('.')[0]
        # #######################
        targets = targets[0]
        samples = samples.to(device)
        outputs = model(samples)

        loss_dict = criterion(outputs, targets, gt_index)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            logger.critical("Loss is {}, stopping training".format(loss_value))
            logger.critical(loss_dict_reduced)
            sys.exit(1)

        # import ipdb; ipdb.set_trace()
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # ###################################
        src_masks = outputs["pred_masks"]
        src_masks = utils.interpolate(
            src_masks, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        gt = gt_index[0]
        src_masks_tmp = src_masks[:, gt].squeeze()  # get the center frame
        src_masks_tmp = src_masks_tmp.argmax(0)
        yc = src_masks_tmp.squeeze(0)
        # import ipdb;ipdb.set_trace()
        yc_one_hot = torch.nn.functional.one_hot(
            yc.reshape(-1).to(torch.int64).cuda(), num_classes=80)
        # ########################################
        targets_tmp = targets
        targets_tmp = torch.nn.functional.one_hot(
            targets_tmp.squeeze().reshape(-1).to(torch.int64).cuda(), num_classes=80)
        iou_tmp = metric.new_iou(targets_tmp, yc_one_hot, 80)
        iou = iou_tmp.cpu().numpy()
        #########################################
        # import ipdb; ipdb.set_trace()
        if running_video_name is None or running_video_name != video_name:
            running_video_name = video_name
            iou_dict[running_video_name] = []
            # import ipdb; ipdb.set_trace()
        iou_dict[running_video_name].append(iou)
        if save_masks:
            # import ipdb;ipdb.set_trace()
            yc_numpy = yc.cpu().numpy()
            name = '-'.join([metadata.classes[class_id] for class_id in np.unique(yc_numpy)[1:]])
            palette = metadata.palette
            color_mask = palette[yc_numpy]  # get_dataset_colormap.label_to_color_image(label=, dataset='ade20k')
            vid_out_dir_cm = os.path.join(output_viz_dir, 'color_mask', running_video_name)
            if not os.path.exists(vid_out_dir_cm):
                os.makedirs(vid_out_dir_cm)
            cv2.imwrite(os.path.join(vid_out_dir_cm, '%s_%s_%0.3f.png' % (target_frame_name, name, iou)), color_mask)
            # ########################################
            target_im_np = cv2.imread(target_im_path[0])
            target_im = Image.fromarray(np.uint8(target_im_np)).convert('RGBA')
            overlay = Image.fromarray(np.uint8(color_mask)).convert('RGBA')
            # import ipdb; ipdb.set_trace()
            blended_arr = Image.blend(target_im, overlay, 0.8).convert('RGB')
            blended_arr = PIL2array(blended_arr, 3)
            blended_img = target_im_np.copy()
            blended_img[yc_numpy > 0, :] = blended_arr[yc_numpy > 0, :]
            vid_out_dir_ol = os.path.join(output_viz_dir, 'pred_overlay', running_video_name)
            if not os.path.exists(vid_out_dir_ol):
                os.makedirs(vid_out_dir_ol)
            cv2.imwrite(os.path.join(vid_out_dir_ol, '%s_%s_%0.3f.png' % (target_frame_name, name, iou)),
                        np.asarray(blended_img))
        if save_gt:
            # import ipdb;ipdb.set_trace()
            gt_numpy = targets.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            name = '-'.join([metadata.classes[class_id] for class_id in np.unique(gt_numpy)[1:]])
            palette = metadata.palette
            color_mask_gt = palette[gt_numpy]  # get_dataset_colormap.label_to_color_image(label=, dataset='ade20k')
            target_im_np = cv2.imread(target_im_path[0])
            target_im = Image.fromarray(np.uint8(target_im_np)).convert('RGBA')
            color_mask_gt = Image.fromarray(np.uint8(color_mask_gt)).convert('RGBA')
            blended_img = Image.blend(target_im, color_mask_gt, 0.8).convert('RGB')
            blended_arr = PIL2array(blended_img, 3)
            overlay_img = target_im_np.copy()
            overlay_img[gt_numpy > 0, :] = blended_arr[gt_numpy > 0, :]
            vid_out_dir_ol = os.path.join(output_viz_dir, 'gt_overlay', running_video_name)
            if not os.path.exists(vid_out_dir_ol):
                os.makedirs(vid_out_dir_ol)
            cv2.imwrite(os.path.join(vid_out_dir_ol, '%s_%s.png' % (target_frame_name, name)), np.asarray(overlay_img))
    mean_iou = np.mean([np.mean(vid_iou_f) for vid_name, vid_iou_f in iou_dict.items()])
    logger.debug('Test results summary--------------------------------------')
    logger.debug('Epoch:%03d Test mean iou: %0.3f' % (epoch, mean_iou))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.debug("Averaged stats:{}".format(metric_logger))
    video_ious = [np.mean(vid_iou_f) for kk, vid_iou_f in iou_dict.items()]
    if verbose:
        for kk, vid_iou_f in iou_dict.items():
            logger.debug('%s -------- %0.3f' % (kk, np.mean(vid_iou_f)))
    return mean_iou, video_ious
