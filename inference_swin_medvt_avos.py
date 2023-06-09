"""
Inference code for MED-VT.
Based on VisTR (https://github.com/Epiphqny/VisTR)
and DETR (https://github.com/facebookresearch/detr)
"""
import time
import argparse
import logging
import random
import numpy as np
import os
import torch

import avos.utils.misc as utils_misc
from avos.evals import run_inference
from avos.evals import inference_on_all_vos_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args_parser():
    parser = argparse.ArgumentParser('MED-VT', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='swinB', type=str,
                        help="backbone to use, [swinS, swinB]")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=(6, 1), type=tuple,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--encoder_cross_layer', default=True, type=bool,
                        help="Cross resolution attention")
    parser.add_argument('--dec_layers', default=9, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dec_multiscale', default='yes', type=str,
                        help="Multi-scale vs single scale decoder, for ablation")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_frames', default=6, type=int,
                        help="Number of frames")
    parser.add_argument('--num_queries', default=6, type=int,
                        help="Number of query slots")
    parser.add_argument('--val_size', default=473, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Label Propagator
    parser.add_argument('--lprop_mode', default=2, type=int, help='no_lprop:0; unidirectional: 1;  bidir:2 ')
    parser.add_argument('--lprop_scale', default=8.0, type=float, help='default 16; use less to fit gpu memory')
    parser.add_argument('--feat_loc', default='late', type=str, help='early or late ')
    parser.add_argument('--stacked_lprop', type=int, default=1, help="repeat the lprop")
    parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER, help='for two-stage train')
    parser.add_argument('--fine_tune_lprop', default=False, action='store_true')

    # Init Weights
    parser.add_argument('--is_train', default=0, type=int,
                             help='Choose 1 for train')
    parser.add_argument('--model_path', type=str,
                        default='./ckpts/swin_medvt/swin_medvt.pth', required=True,
                        help="Path to the model weights.")
    parser.add_argument('--swin_b_pretrained_path', type=str,
                        default="./ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                        help="swin-s pretrained model path.")

    # LOSS
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)

    # Segmentation
    parser.add_argument("--save_pred", action="store_true", default=True)
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--sequence_names', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--msc', action='store_true')
    parser.add_argument('--flip', action='store_true')

    # Misc
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    device = torch.device(args.device)
    utils_misc.init_distributed_mode(args)
    seed = args.seed + utils_misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args.aux_loss = 0
    args.aux_loss_norm = 0
    from avos.models.medvt_swin import build_model_medvt_swinbackbone as build_model
    model, _ = build_model(args)
    model.to(device)
    args.sequence_names = None
    if args.dataset == 'all':
        inference_on_all_vos_dataset(args, device, model)
    else:
        run_inference(args, device, model)
    print('Thank You!')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('VisVOS inference script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    if not hasattr(parsed_args, 'output_dir') or parsed_args.output_dir is None or len(parsed_args.output_dir) < 3:
        from avos.evals import create_eval_save_dir_name_from_args
        out_dir_name = create_eval_save_dir_name_from_args(parsed_args)
        parsed_args.output_dir = os.path.join(os.path.dirname(parsed_args.model_path), out_dir_name)
    if not os.path.exists(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)
    experiment_name = str(parsed_args.model_path).split('/')[-2]
    print('output_dir: '+str(parsed_args.output_dir))
    print('experiment_name:%s' % experiment_name)
    print('log file: ' + str(os.path.join(parsed_args.output_dir, 'out.log')))  # added by @RK
    logging.basicConfig(
        filename=os.path.join(parsed_args.output_dir, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger.debug(parsed_args)
    main(parsed_args)
