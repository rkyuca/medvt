import sys
sys.path.append('./')
import os
import time
import random
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import mask_iou
from utils.system import setup_logging

# import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
"""
python train_medvt.py --session_name s4_pvt_bidir --visual_backbone pvt --train_batch_size 2 --lr 0.0001 --mm_context bidir 
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
    parser.add_argument('--swin_b_pretrained_path', type=str,
                        default="/local/riemann/home/rezaul/projects/medvt2w/ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                        help="swin-s pretrained model path.")

    # TPAVI
    parser.add_argument('--sa_loss_flag', action='store_true', default=False,
                        help='additional loss for last four frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--sa_loss_stages", default=[], nargs='+', type=int,
                        help='compute sa loss in which stages: [0, 1, 2, 3')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')

    parser.add_argument("--tpavi_stages", default=[0, 1, 2, 3],
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
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--train_im_size", default=384, type=int)
    parser.add_argument("--val_im_size", default=384, type=int)
    parser.add_argument("--val_batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--end_lr', default=1e-6, type=float)
    parser.add_argument('--lr_drop', default=4, type=int)
    parser.add_argument('--poly_power', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='/local/riemann/home/rezaul/outputs/avsbench/avs_s4/train_logs', type=str)

    args = parser.parse_args()
    from model import MEDVT_AVSModel as AVSModel

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

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
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # for k, v in model.named_parameters():
    #         print(k, v.requires_grad)

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Data
    train_dataset = S4Dataset('train', im_size=args.train_im_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = S4Dataset('val', im_size=args.val_im_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.val_batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

    # Optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "encoder_backbone" not in n and p.requires_grad],
            "lr": args.lr
        },
        {
            "params": [p for n, p in model.named_parameters() if "encoder_backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.max_epoches - 1, end_learning_rate=args.end_lr,
    #                                 power=args.poly_power)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoches)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        model.train()
        lr_scheduler.step()
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask = mask.view(B, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])  # [B*T, 1, 96, 64]
            with torch.no_grad():
                audio_feature = audio_backbone(audio)  # [B*T, 128]

            output, visual_map_list, a_fea_list = model(imgs, audio_feature, batch_size=B,
                                                        num_frames=frame)  # [bs*5, 1, 224, 224]
            loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1).unsqueeze(1),
                                                   a_fea_list, visual_map_list,
                                                   lambda_1=args.lambda_1,
                                                   count_stages=args.sa_loss_stages,
                                                   sa_loss_flag=args.sa_loss_flag,
                                                   mask_pooling_type=args.mask_pooling_type)

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if (global_step - 1) % 1000 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lambda_1:%.4f, lr: %.4f' % (
                    global_step - 1, max_step, avg_meter_total_loss.pop('total_loss'),
                    avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), args.lambda_1,
                    optimizer.param_groups[0]['lr'])
                logger.info(train_log)

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, _, _ = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)
                output, _, _ = model(imgs, audio_feature, batch_size=B, num_frames=frame)  # [bs*5, 1, 224, 224]
                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})
            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth' % (args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)
            miou_list.append(miou)
            max_miou = max(miou_list)
            val_log = 'Epoch: {}, Miou: {}, Best Epoch:{} maxMiou: {}'.format(epoch, miou, best_epoch, max_miou)
            # print(val_log)
            logger.info(val_log)
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
