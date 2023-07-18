"""
Based on
DETR (https://github.com/facebookresearch/detr)
and
VisTR (https://github.com/Epiphqny/VisTR)
"""
import torch
from torch import nn
import torch.nn.functional as F
from avos.utils.misc import interpolate


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


def dice_multiclass_loss(inputs, targets):
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
    inputs = inputs.softmax(1)
    onehot_gt = F.one_hot(targets.long(), num_classes=80)

    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs.t() * onehot_gt).sum(0)
    denominator = inputs.sum(-1) + onehot_gt.sum(0)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.mean()


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


def crossentropy_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
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

    # Re-organize into the right shape
    prob = inputs.unsqueeze(0)
    targets = targets.unsqueeze(0).long()
    C = prob.shape[1]
    prob = prob.permute(0, *range(2, prob.ndim), 1).reshape(-1, C)
    targets = targets.view(-1)

    # Calc the general ce loss
    log_p = F.log_softmax(prob, dim=-1)
    ce = F.nll_loss(log_p, targets, reduction='none')
    # ce = F.cross_entropy(prob, targets, ignore_index=-100)

    # Find the log-prob in the output, regarding the ground-truth label
    all_rows = torch.arange(len(prob))
    log_pt = log_p[all_rows, targets.view(-1)]  # why not use [:, targets.view(-1)]?

    # Revert the probability from log-prob
    pt = log_pt.exp()
    focal_term = (1 - pt) ** gamma

    loss = focal_term * ce

    return loss.mean()


class SetMultiLabelCriterion(nn.Module):
    """ This class computes the loss for our model.
    The code is based on the code from VisTR.
    """

    def __init__(self, weight_dict, losses, aux_loss=0, aux_loss_norm=0):
        """
        Args:
            weight_dict:
            losses:
            aux_loss:
            aux_loss_norm:
        """
        super().__init__()
        self.num_classes = 1
        self.weight_dict = weight_dict
        self.losses = losses
        self.aux_loss = aux_loss
        self.aux_loss_norm = aux_loss_norm != 0

    @staticmethod
    def loss_masks(src_masks, targets, gt_index):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks = [t["masks"] for t in targets]
        target_masks = targets.squeeze()
        # target_masks, valid = nested_tensor_from_tensor_list(target_masks, split=False).decompose()
        target_masks = target_masks.to(src_masks)

        # print('loss_masks->src_masks_shape:%s target_masks_shape:%s'%(str(src_masks.shape), str(target_masks.shape)))
        # import ipdb; ipdb.set_trace()
        num_frames = 1
        num_instances = 1
        if num_instances > 1:
            gt = target_masks[:, 0:num_frames, :, :]
            for i in range(1, num_instances):
                ins_i = target_masks[:, i * num_frames:(i + 1) * num_frames, :, :]
                gt = gt + ins_i
            gt[gt > 0.5] = 1
            target_masks = gt

        # upsample predictions to the target size
        target_size = target_masks.shape[-2:]
        src_masks = interpolate(src_masks, size=target_size, mode="bilinear", align_corners=False)
        focal_loss_ = 0.0
        dice_loss_ = 0.0

        if isinstance(gt_index, list):
            for idx, gt_ind in enumerate(gt_index):
                src_masks_ind = src_masks[:, gt_ind].squeeze()

                src_masks_ind = src_masks_ind.flatten(1)
                target_masks_ind = target_masks[idx].flatten()

                focal_loss_ += crossentropy_focal_loss(src_masks_ind, target_masks_ind) / len(gt_index)
                dice_loss_ += dice_multiclass_loss(src_masks_ind, target_masks_ind) / len(gt_index)
        else:
            src_masks = src_masks[:, gt_index].squeeze()

            src_masks = src_masks.flatten(1)
            target_masks = target_masks.flatten()

            focal_loss_ = crossentropy_focal_loss(src_masks, target_masks)
            # focal_loss_ = focal_loss = torch.hub.load(
            #     'adeelh/pytorch-multi-class-focal-loss',
            #     model='FocalLoss',
            #     gamma=2,
            #     reduction='mean',
            #     force_reload=False
            # )
            # dice_loss_ = dice_multiclass_loss(src_masks, target_masks)
        losses = {
            "loss_mask": focal_loss_,  # + 0.2 * l1_loss,
            # "loss_dice": dice_loss_,
        }
        return losses

    def forward(self, outputs, targets, gt_idex):
        """ This performs the loss computation.
        Parameters:
             gt_idex:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}

        assert "pred_masks" in outputs
        # import ipdb; ipdb.set_trace()
        src_masks = outputs["pred_masks"]  # index to one single frame
        main_losses = self.loss_masks(src_masks, targets, gt_idex)

        if "aux_pred_masks" in outputs and self.aux_loss > 0:
            # import ipdb; ipdb.set_trace()
            aux_pred_masks = outputs["aux_pred_masks"]
            aux_losses = self.loss_masks(aux_pred_masks, targets, gt_idex)
            if self.aux_loss_norm:
                w_main = 1.0 / (1.0 + self.aux_loss)
                w_aux = self.aux_loss / (1.0 + self.aux_loss)
            else:
                w_main = 1.0
                w_aux = self.aux_loss
            keys = main_losses.keys()
            for k in keys:
                main_losses[k] = w_main * main_losses[k] + w_aux * aux_losses[k]

        losses.update(main_losses)
        return losses
