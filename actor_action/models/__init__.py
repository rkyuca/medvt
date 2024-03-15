import torch
import logging

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def build_medvt_model_a2d(args):
    # Model
    assert hasattr(args, 'backbone') and args.backbone is not None
    if args.backbone == 'swinB':
        print('Using backbone SwinB')
        from avos.models.medvt_swin import build_model_medvt_swinbackbone_without_criterion
        model = build_model_medvt_swinbackbone_without_criterion(args)
    elif args.backbone == 'resnet101':
        print('Using backbone resnet101')
        from avos.models.medvt_resnet import build_model_without_criterion
        model = build_model_without_criterion(args)
    else:
        raise ValueError(f'args.backbone: {args.backbone} not implemented. Use swinB or resnet101.')

    # Loss
    from actor_action.models import criterions
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    if args.is_train:
        criterion = criterions.SetMultiLabelCriterion(weight_dict=weight_dict, losses=losses,
                                                      aux_loss=args.aux_loss, aux_loss_norm=args.aux_loss_norm)
    else:
        criterion = criterions.SetMultiLabelCriterion(weight_dict=weight_dict, losses=losses)
    criterion.to(torch.device(args.device))
    return model, criterion
