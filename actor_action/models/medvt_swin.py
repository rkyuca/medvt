"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
import torch
import logging
from avos.models.medvt_swin import build_model_medvt_swinbackbone_without_criterion
from actor_action.models import criterions

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def build_model_swin_medvt(args):
    # Model
    model = build_model_medvt_swinbackbone_without_criterion(args)

    # Loss
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    if args.is_train:
        criterion = criterions.SetMultiLabelCriterion(weight_dict=weight_dict, losses=losses,
                                                      aux_loss=args.aux_loss, aux_loss_norm=args.aux_loss_norm)
    else:
        criterion = criterions.SetMultiLabelCriterion(weight_dict=weight_dict, losses=losses)
    criterion.to(torch.device(args.device))
    return model, criterion
