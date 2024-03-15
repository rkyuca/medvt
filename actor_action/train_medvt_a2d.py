"""
Training script for MED-VT with Swin backbone on A2D
"""
import sys
sys.path.append('./')
import argparse
import csv
import datetime
import time
import re
from pathlib import Path
import math
import os
import sys
from typing import Iterable
import numpy as np
import random
import logging
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn
import avos.utils.misc as avos_utils
from avos.utils.torch_poly_lr_decay import PolynomialLRDecay
from avos.models.utils import parse_argdict
from actor_action.datasets import transforms as T
from actor_action.datasets.train.a2d_data import A2dDataset
from actor_action.datasets.test.a2d_data_test import A2dDatasetTest
from actor_action.datasets import path_config as dataset_path_config
from actor_action.models import build_medvt_model_a2d as build_model
from actor_action.utils.scheduler import GradualWarmupScheduler
from actor_action.evals_a2d import evaluate

cudnn.benchmark = False
cudnn.deterministic = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)

    # Backbone
    parser.add_argument('--backbone', default='swinB', type=str,
                             help="backbone to use, [swinS, swinB, resnet101]")
    parser.add_argument('--dilation', default=[False, False, False], action='store_true',
                             help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                             help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--enc_layers', default=(6, 1), type=tuple,
                             help="Number of encoding layers in the transformer")
    parser.add_argument('--encoder_cross_layer', default=False, type=bool,
                             help="Cross resolution attention")
    parser.add_argument('--dec_layers', default=9, type=int,
                             help="Number of decoding layers in the transformer")
    parser.add_argument('--query_decoder_scales', default=3, type=int,
                        help="Multi-scale vs single scale decoder, for ablation")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                             help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                             help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                             help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                             help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=6, type=int,
                             help="Number of frames")
    parser.add_argument('--num_queries', default=6, type=int,
                             help="Number of query sots")
    parser.add_argument('--decoder_bbox_head', default='frame', type=str,
                        help="tube or frame")
    parser.add_argument('--bbox_nhead', default=8, type=int,
                        help="dec-out nhead")
    parser.add_argument('--pre_norm', action='store_true')

    # Label Propagator
    parser.add_argument('--lprop_mode', default=0, type=int, help='no_lprop:0; unidirectional: 1;  bidir:2 ')
    parser.add_argument('--lprop_scale', default=16.0, type=float, help='default 16; use less to fit gpu memory')
    parser.add_argument('--feat_loc', default='late', type=str, help='early or late ')
    parser.add_argument('--stacked_lprop', type=int, default=1, help="repeat the lprop")
    parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER, help='for two-stage train')

    # Segmentation
    parser.add_argument('--masks', default=True, action='store_true',
                             help="Train segmentation head if the flag is provided")
    parser.add_argument('--num_classes', default=80, type=int,
                             help="Train segmentation head if the flag is provided")

    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)

    # Train Params
    parser.add_argument('--is_train', default=1, type=int,
                             help='Choose 1 for train')
    # parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--end_lr', default=1e-6, type=float)
    parser.add_argument('--use_lr_warmup', action="store_true")
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--scheduler_type', default='poly', type=str)
    parser.add_argument('--poly_power', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--aux_loss', default=0.5, type=float)
    parser.add_argument('--aux_loss_norm', default=0, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument(
        '--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--batch_size', default=1, type=int)

    # Dataset Paths
    parser.add_argument(
        '--a2d_csv_path', default=dataset_path_config.a2d_csv_path, type=str)
    parser.add_argument(
        '--a2d_frame_path', default=dataset_path_config.a2d_frame_path, type=str)
    parser.add_argument(
        '--a2d_gt_path', default=dataset_path_config.a2d_gt_path, type=str)
    
    # Init Paths
    # Initialize backbone
    parser.add_argument('--resnet101_coco_weights_path', type=str,
                        default="/local/riemann/home/rezaul/model_zoo/medvt2w/ckpts/resnet_init/384_coco_r101.pth",
                        help="Path to the pretrained model.")
    parser.add_argument('--swin_b_pretrained_path', type=str,
                        default="/local/riemann/home/rezaul/model_zoo/medvt2w/ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                        help="swin-s pretrained model path.")
    parser.add_argument('--pvt_weights_path', type=str,
                        default="/local/riemann/home/rezaul/model_zoo/pvt/pvt_v2_b5.pth",
                        help="pvt pretrained model path.")

    """
    parser.add_argument('--pretrained_weights', type=str,
                             default="./ckpts/swin_init/384_coco_r101.pth",
                             help="Path to the pretrained model.")
    parser.add_argument('--swin_s_pretrained_path', type=str,
                             default="./ckpts/swin_init/swin_small_patch244_window877_kinetics400_1k.pth",
                             help="swin-s pretrained model path.")
    parser.add_argument('--swin_b_pretrained_path', type=str,
                             default="./ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                             help="swin-s pretrained model path.")
    parser.add_argument('--resnet101_coco_weights_path', type=str,
                             default="./ckpts/resnet_init/384_coco_r101.pth",
                             help="Path to the pretrained model.")
    """

    # Save Paths
    parser.add_argument('--output_dir', default='/local/riemann1/home/rezaul/outputs/medvtmm/actor_action/train/',
                             help='path where to save, empty for no saving')
    parser.add_argument('--experiment_name', default='medvt_a2d_exp11a2r_20230627_{params_summary}')
    parser.add_argument('--device', default='cuda',
                             help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=False,
                             help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                             help='start epoch')
    parser.add_argument('--eval', action='store_true')
 
    # Misc
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--vis_freq', type=int, default=1000)
    parser.add_argument('--world_size', default=1, type=int,
                             help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                            help='url used to set up distributed training')
    return parser


def record_csv(filepath, row):
    with open(filepath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    output_viz_dir=Path('./outputs/'), total_epochs=15, vis_freq: int = 1000):
    inverse_norm_transform = T.InverseNormalizeTransforms()
    model.train()
    criterion.train()
    metric_logger = avos_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', avos_utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}/{}]:'.format(epoch, total_epochs)
    print_freq = 500
    i_iter = 0
    if not os.path.exists(output_viz_dir):
        os.makedirs(output_viz_dir)
    _loss_t_csv_fn = os.path.join(output_viz_dir, 'loss.csv')
    if epoch == 0 and os.path.exists(_loss_t_csv_fn):
        os.rename(_loss_t_csv_fn, os.path.join(
            output_viz_dir, 'loss_{}.csv'.format(time.time())))
    loss_sum = 0
    item_count = 0
    tt1 = time.time()
    for samples, vid_cls, targets, vid_id, gt_index in metric_logger.log_every(data_loader, print_freq, header):
        i_iter = i_iter + 1
        if i_iter % print_freq == 0:
            tt2 = time.time()
            dt_tm = str(datetime.timedelta(seconds=int(tt2 - tt1)))
            logger.debug('{} i_iter_Training_time:{} '.format(i_iter, dt_tm))
            # import ipdb; ipdb.set_trace()
        samples = samples.to(device)
        targets = targets[0]
        gt_index = gt_index[0]
        # samples = nested_tensor_from_tensor_list([samples])
        # targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets, gt_index)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = avos_utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            logger.critical("Loss is {}, stopping training".format(loss_value))
            logger.critical(loss_dict_reduced)
            logger.debug('video_name: {} frame_ids:{} center_frame:{}'.format(targets[0]['video_name'],
                                                                              str(
                                                                                  targets[0]['frame_ids']),
                                                                              targets[0]['center_frame']))
            sys.exit(1)
        optimizer.zero_grad()
        losses = losses * (math.exp(-(epoch / 10)))
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        loss_sum += float(loss_value)
        item_count += 1
        if i_iter % 50 == 49:
            loss_avg = loss_sum / item_count
            loss_sum = 0
            item_count = 0
            record_csv(_loss_t_csv_fn, ['%e' % loss_avg])
    metric_logger.synchronize_between_processes()
    logger.debug("Averaged stats:{}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    logger.debug('starting main ...')
    avos_utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + avos_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ########################### ####################
    # ### DATASETS ###################################
    dataset_train = A2dDataset(csv_path=args.a2d_csv_path,
                               frame_path=args.a2d_frame_path,
                               gt_path=args.a2d_gt_path,
                               context_len=int(args.num_frames / 2),
                               # context_len=2, # for inter-frame strategy
                               train=True)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    if args.debug:
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       num_workers=args.num_workers)
    else:
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=avos_utils.collate_fn, num_workers=args.num_workers)
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
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=avos_utils.collate_fn,
                                 num_workers=args.num_workers)
    # ########### MODEL #################################
    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": args.lr
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler_type == 'poly':
        lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs - 1 - args.warmup_epochs,
                                         end_learning_rate=args.end_lr, power=args.poly_power)
    elif args.scheduler_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    elif args.scheduler_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs)

    if args.use_lr_warmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs,
                                                  after_scheduler=lr_scheduler)
    # ############################################################################
    output_dir = Path(args.output_dir)
    output_viz_dir = output_dir / 'viz'
    # import ipdb; ipdb.set_trace()
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            logger.debug('start_epoch:{}'.format(args.start_epoch))
    start_time = time.time()
    best_eval_iou = 0
    best_eval_epoch = 0
    logger.debug("Training ...")
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if not args.use_lr_warmup:
            if epoch == 0:
                optimizer.param_groups[1]['lr'] = 0
            elif epoch == 1:
                optimizer.param_groups[1]['lr'] = args.lr_backbone
        logger.debug('epoch: %3d  optimizer.param_groups[0][lr]: %e' % (
            epoch, optimizer.param_groups[0]['lr']))
        logger.debug('epoch: %3d  optimizer.param_groups[1][lr]: %e' % (
            epoch, optimizer.param_groups[1]['lr']))
        train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, output_viz_dir, total_epochs=args.epochs, vis_freq=args.vis_freq)
        if args.use_lr_warmup:
            scheduler_warmup.step(epoch=epoch)
        else:
            lr_scheduler.step()
        t2 = time.time()
        # import ipdb; ipdb.set_trace()
        mean_iou, davis_iou_list = evaluate(
            model, criterion, data_loader_val, device, epoch, output_viz_dir)
        logger.debug('**************************')
        logger.debug('**************************')
        logger.debug('[Epoch:%2d] val_mean_iou:%0.3f' % (epoch, mean_iou))
        if mean_iou > best_eval_iou:
            best_eval_iou = mean_iou
            best_eval_epoch = epoch
        logger.debug('Best eval epoch:%03d mean_iou: %0.3f' %(best_eval_epoch, best_eval_iou))
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_last.pth']
            if epoch == best_eval_epoch:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
            for checkpoint_path in checkpoint_paths:
                logger.debug('saving ...checkpoint_path:{}'.format(
                    str(checkpoint_path)))
                avos_utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    }, checkpoint_path)
        t3 = time.time()
        train_time_str = str(datetime.timedelta(seconds=int(t2 - t1)))
        eval_time_str = str(datetime.timedelta(seconds=int(t3 - t2)))
        logger.debug(
                'Epoch:{}/{} Training_time:{} Eval_time:{}'.format(epoch, args.epochs, train_time_str, eval_time_str))
        logger.debug('**************************')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.debug('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(
        'VisTR training and evaluation script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    default_args = args_parser.parse_args([])
    if parsed_args.pretrain_settings is not None:
        parsed_args.pretrain_settings = parse_argdict(parsed_args.pretrain_settings)
    else:
        parsed_args.pretrain_settings = {}
    # ################
    params_summary = '%s_%s_df%d_enc%s_dec%s_t%dv%df%d_lppmode%d_lppsc%0.1f_lr%0.1e_%0.1e_aux%0.1f_ep%02d' % (
        datetime.datetime.today().strftime('%Y%m%d%H%M%S'),
        parsed_args.backbone,
        parsed_args.dim_feedforward,
        str(re.sub('[ |,|\'|(|)]', '', str(parsed_args.enc_layers))),
        str(parsed_args.dec_layers),
        320, 320, parsed_args.num_frames,
        parsed_args.lprop_mode,
        parsed_args.lprop_scale,
        parsed_args.lr, parsed_args.lr_backbone,
        parsed_args.aux_loss,
        parsed_args.epochs
        )
    print('params_summary:%s' % params_summary)
    parsed_args.experiment_name = parsed_args.experiment_name.replace('{params_summary}', params_summary)
    print('parsed_args.experiment_name: %s' % parsed_args.experiment_name)
    # ##########################
    output_path = os.path.join(
        parsed_args.output_dir, parsed_args.experiment_name)
    parsed_args.output_dir = output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(
        filename=os.path.join(output_path, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.debug(parsed_args)
    logger.debug('output_dir: ' + str(output_path))
    logger.debug('experiment_name:%s' % parsed_args.experiment_name)
    logger.debug('log file: ' + str(os.path.join(parsed_args.output_dir, 'out.log')))
    main(parsed_args)
    logger.debug('Finished training...')
