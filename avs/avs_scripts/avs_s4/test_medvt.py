import sys
sys.path.append('./')
import os
import time
import torch
import argparse
import logging

from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish

from utils import pyutils
from utils.utility import mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
python -W ignore avs/avs_scripts/avs_s4/test_medvt.py --session_name s4_pvt  --tpavi_va_flag --weights /local/riemann/home/rezaul/outputs/avsbench/avs_s4/train_logs/s4_pvt_bidir_20240302-103107/checkpoints/s4_pvt_bidir_best.pth
"""

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Backbone
    parser.add_argument("--visual_backbone", default="pvt", type=str,
                        help="use resnet50, pvt, swinB as the visual backbone")
    parser.add_argument("--load_pretrained_backbone", default=1, type=int)
    parser.add_argument('--mm_context', default='bidir', type=str,
                        help="multimodal context to use, [tpavi, bidir, joint]")

    # Initialize backbone
    parser.add_argument('--resnet101_coco_weights_path', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2w/ckpts/resnet_init/384_coco_r101.pth",
                        help="Path to the pretrained model.")
    parser.add_argument('--swin_b_pretrained_path', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2w/ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                        help="swin-s pretrained model path.")

    parser.add_argument('--pvtv2_pretrained_path', type=str,
                        default="/local/riemann/home/rezaul/model_zoo/medvtplus_release/pretrained_backbones/avsbench/pvt_v2_b5.pth",
                        help="PVTv2 pretrained model path.")

    # TPAVI
    parser.add_argument('--sa_loss_flag', action='store_true', default=False,
                        help='additional loss for last four frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--sa_loss_stages", default=[], nargs='+', type=int,
                        help='compute sa loss in which stages: [0, 1, 2, 3')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')

    parser.add_argument("--tpavi_stages", default=[0, 1, 2, 3], nargs='+', type=int,
                        help='add tpavi block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    # Visual Context Encoder
    parser.add_argument("--vce_enc_layer", default=[3, 1], nargs='+', type=int,
                        help='add encoder layers from deeper to shallow order ')
    parser.add_argument("--vce_use_cross_layer", default=0, type=bool,
                        help='use or not use cross scale encoder')
    parser.add_argument("--vce_nhead", default=8, type=int,
                        help='vce_nhead')
    parser.add_argument("--vce_dim_ff", default=2048, type=int,
                        help='vc_dim_ff')

    # Mask decoder @TODO

    # Train/Test
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--test_im_size", default=384, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-6, type=float)
    parser.add_argument('--end_lr', default=1e-6, type=float)
    parser.add_argument('--lr_drop', default=4, type=int)
    parser.add_argument('--poly_power', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument("--num_workers", default=8, type=int)

    # Test
    parser.add_argument("--weights",type=str)
    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='/local/riemann/home/rezaul/outputs/medvtplus_release/avs_s4_pvtv2_medvt', type=str)

    args = parser.parse_args()
    from model import MEDVT_AVSModel as AVSModel


    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(config=cfg, args=args)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    logger.info('=> Load trained model %s'%args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Test data
    # split = 'test'
    test_dataset = S4Dataset('test', im_size = args.test_im_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        drop_last=True)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')
    # Test
    model.eval()
    with torch.no_grad():
        total_iter = len(test_dataloader)
        for n_iter, batch_data in  enumerate(test_dataloader):
            if (n_iter+1)%100==0:
                print('iter %d of %d'%(n_iter, total_iter))
            imgs, audio, mask, category_list, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            # import ipdb;ipdb.set_trace()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            with torch.no_grad():
                audio_feature = audio_backbone(audio)
            output, _, _ = model(imgs, audio_feature) # [5, 1, 224, 224] = [bs=1 * T=5, 1, 224, 224]
            if args.save_pred_mask:
                mask_save_path = os.path.join(log_dir, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, category_list, video_name_list)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou.item()})
            F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
            avg_meter_F.add({'F_score': F_score})
            # print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))
        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou)
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou, F_score))












