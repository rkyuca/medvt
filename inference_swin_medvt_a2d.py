"""
Training script for ResNet+Transformer for VOS
Based on training script of VisTR (https://github.com/Epiphqny/VisTR)
Which was modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import csv
import datetime
import time
from pathlib import Path
import math
import os
import sys
import numpy as np
import random
import logging
import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn

import actor_action.util.misc as utils
from actor_action.datasets.test.a2d_data_test import A2dDatasetTest
from actor_action.datasets import path_config as dataset_path_config
from actor_action.models.medvt_swin import build_model_swin_medvt as build_model
from actor_action.util.wandb_utils import init_or_resume_wandb_run
from actor_action.util import metric
from avos.models.utils import parse_argdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cudnn.benchmark = False
cudnn.deterministic = True


def get_args_parser():
    args_parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)

    args_parser.add_argument('--is_train', default=0, type=int,
                             help='Choose 1 for train')
    args_parser.add_argument('--config', type=str, default='./actor_action/configs/medvt_swin_a2d.yaml')

    # Backbone
    # args_parser.add_argument('--dilation', default=[False, False, False], action='store_true',
    #                          help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    args_parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                             help="Type of positional embedding to use on top of the image features")

    # * Transformer
    args_parser.add_argument('--enc_layers', default=(6, 1), type=tuple,
                             help="Number of encoding layers in the transformer")
    args_parser.add_argument('--encoder_cross_layer', default=False, type=bool,
                             help="Cross resolution attention")
    args_parser.add_argument('--dec_layers', default=9, type=int,
                             help="Number of decoding layers in the transformer")
    args_parser.add_argument('--dec_multiscale', default='yes', type=str,
                             help="Multi-scale vs single scale decoder, for ablation")
    args_parser.add_argument('--dim_feedforward', default=2048, type=int,
                             help="Intermediate size of the feedforward layers in the transformer blocks")
    args_parser.add_argument('--hidden_dim', default=384, type=int,
                             help="Size of the embeddings (dimension of the transformer)")
    args_parser.add_argument('--dropout', default=0.1, type=float,
                             help="Dropout applied in the transformer")
    args_parser.add_argument('--nheads', default=8, type=int,
                             help="Number of attention heads inside the transformer's attentions")
    args_parser.add_argument('--num_frames', default=6, type=int,
                             help="Number of frames")
    args_parser.add_argument('--num_queries', default=6, type=int,
                             help="Number of query sots")
    args_parser.add_argument('--pre_norm', action='store_true')
    args_parser.add_argument('--use_multiscale_enc', action='store_true')
    args_parser.add_argument('--decoder_type', default='', type=str)

    # Label Propagator
    args_parser.add_argument('--lprop_mode', default=2, type=int, help='no_lprop:0; unidirectional: 1;  bidir:2 ')
    args_parser.add_argument('--lprop_scale', default=16.0, type=float, help='default 16; use less to fit gpu memory')
    args_parser.add_argument('--feat_loc', default='late', type=str, help='early or late ')
    args_parser.add_argument('--stacked_lprop', type=int, default=1, help="repeat the lprop")
    args_parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER, help='for two-stage train')

    # Init Weights
    args_parser.add_argument('--backbone', default='swinB', type=str,
                             help="backbone to use, [swinS, swinB]")
    args_parser.add_argument('--model_path', type=str, default='./ckpts/swin_medvt_a2d/swin_medvt_a2d.pth',
                             help="Path to the model weights.")
    args_parser.add_argument('--swin_b_pretrained_path', type=str,
                             default="./ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                             help="swin-s pretrained model path.")
    args_parser.add_argument('--resnet101_coco_weights_path', type=str,
                             default="./ckpts/resnet_init/384_coco_r101.pth",
                             help="Path to the pretrained model.")

    # Segmentation
    args_parser.add_argument('--masks', default=True, action='store_true',
                             help="Train segmentation head if the flag is provided")
    args_parser.add_argument('--num_classes', default=80, type=int,
                             help="Train segmentation head if the flag is provided")
    args_parser.add_argument('--output_dir', default='./outputs/swin_medvt_a2d',
                             help='path where to save, empty for no saving')

    # * Loss coefficients
    args_parser.add_argument('--aux_loss', default=0, type=float)
    args_parser.add_argument('--aux_loss_norm', default=0, type=float)
    args_parser.add_argument('--mask_loss_coef', default=1, type=float)
    args_parser.add_argument('--dice_loss_coef', default=1, type=float)

    # Datasets
    args_parser.add_argument(
        '--a2d_csv_path', default=dataset_path_config.a2d_csv_path, type=str)
    args_parser.add_argument(
        '--a2d_frame_path', default=dataset_path_config.a2d_frame_path, type=str)
    args_parser.add_argument(
        '--a2d_gt_path', default=dataset_path_config.a2d_gt_path, type=str)

    # Misc
    # args_parser.add_argument(
    #    '--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    args_parser.add_argument('--batch_size', default=1, type=int)
    args_parser.add_argument('--train_size', default=392, type=int)
    args_parser.add_argument('--val_size', default=360, type=int)
    args_parser.add_argument('--experiment_name', default='medvt')
    args_parser.add_argument('--use_wandb', action='store_true')
    args_parser.add_argument('--wandb_user', type=str, default='yvv')
    args_parser.add_argument(
        '--wandb_project', type=str, default='transformervos')
    args_parser.add_argument('--vis_freq', type=int, default=5)
    args_parser.add_argument('--device', default='cuda',
                             help='device to use for training / testing')
    args_parser.add_argument('--seed', default=42, type=int)
    args_parser.add_argument('--num_workers', default=4, type=int)
    args_parser.add_argument('--world_size', default=1, type=int,
                             help='number of distributed processes')
    args_parser.add_argument(
        '--dist_url', default='env://', help='url used to set up distributed training')


    return args_parser


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


def main(args):
    print('starting main ...')
    # import ipdb; ipdb.set_trace()
    utils.init_distributed_mode(args)
    logger.debug("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ########################### ####################
    # ### DATASETS ###################################
    dataset_val = A2dDatasetTest(csv_path=args.a2d_csv_path,
                                 frame_path=args.a2d_frame_path,
                                 gt_path=args.a2d_gt_path,
                                 context_len=int(args.num_frames / 2),
                                 # context_len=2,
                                 train=False)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_val, args.batch_size, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)
    # ########### MODEL #################################
    # import ipdb;ipdb.set_trace()
    model, criterion = build_model(args)
    model.to(device)

    # import ipdb; ipdb.set_trace()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.debug('number of params:{}'.format(n_parameters))

    # ############################################################################
    output_dir = Path(args.output_dir)
    output_viz_dir = output_dir / 'viz'
    # import ipdb; ipdb.set_trace()
    # load coco pretrained weight
    checkpoint = torch.load(args.model_path, map_location='cpu')['model']
    if args.distributed:
        model.module.load_state_dict(checkpoint, strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    # import ipdb; ipdb.set_trace()
    start_time = time.time()
    # import ipdb;ipdb.set_trace()
    logger.debug("Start evaluation...")
    if True:
        t2 = time.time()
        mean_iou, iou_list = evaluate(
            model, criterion, data_loader_val, device, 0, output_viz_dir,
            use_wandb=args.use_wandb
        )
        logger.debug('**************************')
        logger.debug('val_mean_iou:%0.3f' % mean_iou)
        t3 = time.time()
        eval_time_str = str(datetime.timedelta(seconds=int(t3 - t2)))
        logger.debug('Eval_time:{}'.format(eval_time_str))
        logger.debug(
            '##########################################################')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'VisTR training and evaluation script', parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    default_args = parser.parse_args([])
    if parsed_args.config is not None:
        parsed_args = utils.merge_cfg_args(parsed_args, default_args)

    if parsed_args.pretrain_settings is not None:
        parsed_args.pretrain_settings = parse_argdict(parsed_args.pretrain_settings)
    else:
        parsed_args.pretrain_settings = {}

    output_path = os.path.join(
        parsed_args.output_dir, parsed_args.experiment_name+'_eval')
    print(output_path)
    parsed_args.output_dir = output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('log file: ' + str(os.path.join(output_path, 'out.log')))  # added by @RK
    logging.basicConfig(
        filename=os.path.join(output_path, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger.debug(output_path)
    logger.debug('experiment_name: {}'.format(parsed_args.experiment_name))
    logger.debug(
        "This is our baseline with resnet101s32 and transformer encoder-decoder.")
    logger.debug('Using flip, poly lr, epochs 15, adamw')
    logger.debug("Used Args are {}".format(str(parsed_args)))
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)

    if parsed_args.use_wandb:
        wandb_id_file_path = Path(os.path.join(
            output_path, parsed_args.experiment_name + '_wandb.txt'))
        config = init_or_resume_wandb_run(wandb_id_file_path,
                                          entity_name=parsed_args.wandb_user,
                                          project_name=parsed_args.wandb_project,
                                          run_name=parsed_args.experiment_name,
                                          config=parsed_args)
        logger.debug("Initialized Wandb")
    logger.debug('output_path:'+output_path)
    logger.debug('log file: ' + str(os.path.join(output_path, 'out.log')))  # added by @RK
    main(parsed_args)
    logger.debug('Finished eval...')
