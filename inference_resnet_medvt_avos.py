"""
Inference code for MED-VT.
Based on VisTR (https://github.com/Epiphqny/VisTR)
and DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import logging
import random
import numpy as np
import os
import torch
from avos.utils import misc as misc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def get_args_parser():
    parser = argparse.ArgumentParser('MED-VT', add_help=False)

    # Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=[False, False, False], action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--enc_layers', default=7, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=9, type=int,
                        help="Number of decoding layers in the transformer")
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
    parser.add_argument('--pre_norm', action='store_true')

    # Label Propagator
    parser.add_argument('--lprop_mode', default=2, type=int, help='no_lprop:0; unidirectional: 1;  bidir:2 ')
    parser.add_argument('--lprop_scale', default=8.0, type=float, help='default 16; use less to fit gpu memory')
    parser.add_argument('--feat_loc', default='late', type=str, help='early or late ')
    parser.add_argument('--stacked_lprop', type=int, default=1, help="repeat the lprop")
    parser.add_argument('--temporal_strides', default=[1], nargs='+', type=int,
                        help="temporal strides used to construct input cip")
    parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER, help='for two-stage train')

    # Init Weights
    parser.add_argument('--is_train', default=0, type=int, help='Choose 1 for train')
    parser.add_argument('--model_path', type=str,
                        default='./ckpts/resnet_medvt/resnet_medvt.pth', required=True,
                        help="Path to the model weights.")
    parser.add_argument('--resnet101_coco_weights_path', type=str,
                             default="./ckpts/resnet_init/384_coco_r101.pth",
                             help="Path to the pretrained model.")

    # LOSS
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)

    # Segmentation
    parser.add_argument('--output_dir', required=True,
                        help='path where to save, empty for no saving')
    parser.add_argument("--save_pred", action="store_true", default=True)
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--sequence_names', type=str, default=None)
    parser.add_argument('--val_size', default=473, type=int,
                        help="Number of query slots")
    parser.add_argument('--davis_msc_scales', type=list, default=[0.75, 0.8, 0.9, 1, 1.1, 1.15])  # changed to fit gpu
    parser.add_argument('--davis_input_max_sc', type=float, default=1.65)  # To fit gpu memory, reduced this value
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
    # print(args)
    device = torch.device(args.device)
    misc.init_distributed_mode(args)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ####################################################################
    from avos.models.medvt_resnet import build_model
    model, criterion = build_model(args)
    model.to(device)
    #################################################################
    if args.dataset == 'all':
        from avos.evals import inference_on_all_vos_dataset
        inference_on_all_vos_dataset(args, device, model)
    else:
        from avos.evals import run_inference
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
    """        
    parsed_args.aux_loss = 0
    parsed_args.aux_loss_norm = 0
    if not os.path.exists(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)
    experiment_name = str(parsed_args.model_path).split('/')[-2]
    """
    print('output_dir: ' + str(parsed_args.output_dir))
    print('experiment_name:%s' % experiment_name)
    print('log file: ' + str(os.path.join(parsed_args.output_dir, 'out.log')))  # added by @RK
    logging.basicConfig(
        filename=os.path.join(parsed_args.output_dir, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # logger.debug(parsed_args)
    main(parsed_args)
