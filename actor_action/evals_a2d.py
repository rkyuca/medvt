import sys
import numpy as np
import logging
import torch
import math
import wandb
import torch.backends.cudnn as cudnn
import avos.utils.misc as utils
from actor_action.utils import metric

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cudnn.benchmark = False
cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch: int, output_viz_dir=None,
             total_epochs=15, use_wandb=False):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test Epoch: [{}/{}]:'.format(epoch, total_epochs)
    i_iter = 0
    running_video_name = None
    iou_dict = {}
    for samples, vid_cls, targets, vid_id, gt_index in metric_logger.log_every(data_loader, 500, header):
        i_iter = i_iter + 1
        video_name = vid_id
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
        yc = torch.nn.functional.one_hot(
            yc.reshape(-1).to(torch.int64).cuda(), num_classes=80)
        # ########################################
        out = yc
        targets_tmp = targets
        targets_tmp = torch.nn.functional.one_hot(
            targets_tmp.squeeze().reshape(-1).to(torch.int64).cuda(), num_classes=80)
        iou_tmp = metric.new_iou(targets_tmp, out, 80)
        iou = iou_tmp.cpu().numpy()
        if use_wandb:
            wandb_dict = {'val_loss': loss_value, 'val_iou': iou}
            wandb.log(wandb_dict)
        #########################################
        # import ipdb; ipdb.set_trace()
        if running_video_name is None or running_video_name != video_name:
            running_video_name = video_name
            iou_dict[running_video_name] = {}
            # import ipdb; ipdb.set_trace()
        iou_dict[running_video_name][0] = iou
    mean_iou = np.mean([np.mean(list(vid_iou_f.values()))
                        for _, vid_iou_f in iou_dict.items()])
    logger.debug('Test results summary--------------------------------------')
    logger.debug('Epoch:%03d Test mean iou: %0.3f' % (epoch, mean_iou))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.debug("Averaged stats:{}".format(metric_logger))
    video_ious = [np.mean(list(vid_iou_f.values()))
                  for _, vid_iou_f in iou_dict.items()]
    return mean_iou, video_ious