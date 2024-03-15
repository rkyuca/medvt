"""
Taken from https://github.com/Epiphqny/VisTR
which was released under the Apache 2.0 license.
And modified as needed.
"""
"""
VisTR criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from avos.utils.misc import (nested_tensor_from_tensor_list, interpolate)


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # import ipdb; ipdb.set_trace()
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum()
    # return 5*torch.log(torch.cosh(loss)).sum()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # import ipdb; ipdb.set_trace()
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


class SetCriterion(nn.Module):
    """ This class computes the loss for our model.
    The code is based on the code from VisTR.
    """

    def __init__(self, weight_dict, losses, aux_loss=0, aux_loss_norm=0):
        super().__init__()
        self.num_classes = 1
        self.weight_dict = weight_dict
        self.losses = losses
        self.aux_loss = aux_loss
        self.aux_loss_norm = aux_loss_norm != 0
        self.L1Loss = torch.nn.L1Loss()

    def loss_masks(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        target_masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(target_masks, split=False).decompose()
        target_masks = target_masks.to(src_masks)
        num_frames = src_masks.shape[1]
        num_instances = target_masks.shape[1] // num_frames
        if num_instances > 1:
            gt = target_masks[:, 0:num_frames, :, :]
            for i in range(1, num_instances):
                ins_i = target_masks[:, i * num_frames:(i + 1) * num_frames, :, :]
                gt = gt + ins_i
            gt[gt > 0.5] = 1
            target_masks = gt
        target_size = target_masks.shape[-2:]
        src_masks = interpolate(src_masks, size=target_size, mode="bilinear", align_corners=False)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        focal_loss_ = sigmoid_focal_loss(src_masks, target_masks)
        dice_loss_ = dice_loss(src_masks, target_masks)
        if "aux_pred_masks" in outputs and self.aux_loss>0:
            aux_predictions = outputs["aux_pred_masks"]
            if not type(aux_predictions) is list:
                aux_predictions = [aux_predictions]
            if self.aux_loss_norm:
                w_main = 1.0/(1.0 + self.aux_loss)
                w_aux = self.aux_loss/(1.0 + self.aux_loss)
            else:
                w_main = 1.0
                w_aux = self.aux_loss
            focal_loss_ = focal_loss_ * w_main
            dice_loss_ = dice_loss_ * w_main
            for i, aux_pred in enumerate(aux_predictions):
                aux_pred = interpolate(aux_pred, size=target_size, mode="bilinear", align_corners=False)
                aux_pred = aux_pred.flatten(1)
                aux_focal_loss_i = sigmoid_focal_loss(aux_pred, target_masks)
                aux_dice_loss_i = dice_loss(aux_pred, target_masks)
                focal_loss_ = focal_loss_ + aux_focal_loss_i * w_aux
                dice_loss_ = dice_loss_ + aux_dice_loss_i * w_aux
        losses = {
            "loss_mask": focal_loss_,  # + 0.2 * l1_loss,
            "loss_dice": dice_loss_,
        }
        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        losses.update(self.loss_masks(outputs, targets))
        return losses
