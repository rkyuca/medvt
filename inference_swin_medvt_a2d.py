import argparse
import csv
import sys
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
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn
import avos.utils.misc as avos_utils
from actor_action.datasets.test.a2d_data_test import A2dDatasetTest
from actor_action.datasets import path_config as dataset_path_config
from actor_action.models.medvt_swin import build_model_swin_medvt as build_model
from actor_action.utils.wandb_utils import init_or_resume_wandb_run
from avos.models.utils import parse_argdict
from actor_action.evals_a2d import evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cudnn.benchmark = False
cudnn.deterministic = True


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)

    parser.add_argument('--is_train', default=0, type=int,
                             help='Choose 1 for train')
    # Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                             help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--enc_layers', default=(6, 1), type=tuple,
                             help="Number of encoding layers in the transformer")
    parser.add_argument('--encoder_cross_layer', default=False, type=bool,
                             help="Cross resolution attention")
    parser.add_argument('--dec_layers', default=9, type=int,
                             help="Number of decoding layers in the transformer")
    parser.add_argument('--dec_multiscale', default='yes', type=str,
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
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--use_multiscale_enc', action='store_true')
    parser.add_argument('--decoder_type', default='', type=str)

    # Label Propagator
    parser.add_argument('--lprop_mode', default=2, type=int, help='no_lprop:0; unidirectional: 1;  bidir:2 ')
    parser.add_argument('--lprop_scale', default=16.0, type=float, help='default 16; use less to fit gpu memory')
    parser.add_argument('--feat_loc', default='late', type=str, help='early or late ')
    parser.add_argument('--stacked_lprop', type=int, default=1, help="repeat the lprop")
    parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER, help='for two-stage train')

    # Init Weights
    parser.add_argument('--backbone', default='swinB', type=str,
                             help="backbone to use, [swinS, swinB]")
    parser.add_argument('--model_path', type=str, default='./ckpts/swin_medvt_a2d/swin_medvt_a2d.pth',
                             help="Path to the model weights.")
    parser.add_argument('--swin_b_pretrained_path', type=str,
                             default="./ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                             help="swin-s pretrained model path.")
    parser.add_argument('--resnet101_coco_weights_path', type=str,
                             default="./ckpts/resnet_init/384_coco_r101.pth",
                             help="Path to the pretrained model.")

    # Segmentation
    parser.add_argument('--masks', default=True, action='store_true',
                             help="Train segmentation head if the flag is provided")
    parser.add_argument('--num_classes', default=80, type=int,
                             help="Train segmentation head if the flag is provided")
    parser.add_argument('--output_dir', default='./outputs/swin_medvt_a2d',
                             help='path where to save, empty for no saving')

    # Loss coefficients
    parser.add_argument('--aux_loss', default=0, type=float)
    parser.add_argument('--aux_loss_norm', default=0, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)

    # Datasets
    parser.add_argument(
        '--a2d_csv_path', default=dataset_path_config.a2d_csv_path, type=str)
    parser.add_argument(
        '--a2d_frame_path', default=dataset_path_config.a2d_frame_path, type=str)
    parser.add_argument(
        '--a2d_gt_path', default=dataset_path_config.a2d_gt_path, type=str)

    # Misc
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--train_size', default=392, type=int)
    parser.add_argument('--val_size', default=360, type=int)
    parser.add_argument('--experiment_name', default='medvt')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_user', type=str, default='yvv')
    parser.add_argument(
        '--wandb_project', type=str, default='transformervos')
    parser.add_argument('--vis_freq', type=int, default=5)
    parser.add_argument('--device', default='cuda',
                             help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                             help='number of distributed processes')
    parser.add_argument(
        '--dist_url', default='env://', help='url used to set up distributed training')
    return parser


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
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    # logger.debug('number of params:{}'.format(n_parameters))
    # ############################################################################
    output_dir = Path(args.output_dir)
    output_viz_dir = output_dir / 'viz'
    checkpoint = torch.load(args.model_path, map_location='cpu')['model']
    if args.distributed:
        model.module.load_state_dict(checkpoint, strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    start_time = time.time()
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
    args_parser = argparse.ArgumentParser(
        'VisTR training and evaluation script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    default_args = args_parser.parse_args([])

    if parsed_args.pretrain_settings is not None:
        parsed_args.pretrain_settings = parse_argdict(parsed_args.pretrain_settings)
    else:
        parsed_args.pretrain_settings = {}

    output_path = os.path.join(
        parsed_args.output_dir, parsed_args.experiment_name+'_eval')
    parsed_args.output_dir = output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(
        filename=os.path.join(output_path, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.debug('output_path:'+output_path)
    logger.debug('experiment_name: {}'.format(parsed_args.experiment_name))
    logger.debug('log_file: ' + str(os.path.join(output_path, 'out.log')))
    if parsed_args.use_wandb:
        wandb_id_file_path = Path(os.path.join(
            output_path, parsed_args.experiment_name + '_wandb.txt'))
        config = init_or_resume_wandb_run(wandb_id_file_path,
                                          entity_name=parsed_args.wandb_user,
                                          project_name=parsed_args.wandb_project,
                                          run_name=parsed_args.experiment_name,
                                          config=parsed_args)
        logger.debug("Initialized Wandb")
    main(parsed_args)
    logger.debug('Finished eval...')
