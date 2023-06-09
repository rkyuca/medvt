"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import logging

from avos.models.medvt_swin import build_swin_b_backbone
from avos.models.medvt_swin import build_swin_s_backbone
from avos.models.medvt_swin import Transformer
from avos.models.position_encoding import build_position_encoding
from actor_action.models import criterions
from actor_action.models.label_propagation import LabelPropagator as LabelPropagator
from actor_action.util.misc import (nested_tensor_from_tensor_list)
from actor_action.util.misc import NestedTensor

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VOS_SwinMEDVT(nn.Module):
    def __init__(self, args, backbone, backbone_dims, hidden_dim, transformer, num_frames, temporal_strides=[1],
                 n_class=80):
        super().__init__()
        self.temporal_strides = temporal_strides
        self.backbone_name = args.backbone
        self.backbone = backbone
        self.num_frames = num_frames
        self.position_embedding = build_position_encoding(args)
        self.transformer = transformer
        if transformer is None:
            self.input_proj = nn.Conv2d(backbone_dims[-1], hidden_dim, kernel_size=1)
        mask_head_in_channels = hidden_dim
        if transformer is not None and transformer.use_decoder and transformer.decoder_multiscale:
            mask_head_in_channels = hidden_dim + transformer.bbox_nhead
        self.insmask_head = nn.Sequential(
            nn.Conv3d(mask_head_in_channels, 384, (1, 3, 3), padding='same', dilation=1),
            nn.GroupNorm(4, 384),
            nn.ReLU(),
            nn.Conv3d(384, 256, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv3d(128, n_class, 1))

    def _divide_by_stride(self, samples: NestedTensor):
        samples_dict = {}
        for it, stride in enumerate(self.temporal_strides):
            start = it * self.num_frames
            end = (it + 1) * self.num_frames
            samples_dict[stride] = NestedTensor(samples.tensors[start:end], samples.mask[start:end])
        return samples_dict

    def forward(self, samples: NestedTensor):
        if self.training:
            return self._forward_one_samples(samples)
        else:
            return self.forward_inference(samples)

    def forward_inference(self, samples: NestedTensor):
        # import ipdb;ipdb.set_trace()
        samples = self._divide_by_stride(samples)
        all_outs = []
        mean_outs = None
        with torch.no_grad():
            for stride, samples_ in samples.items():
                all_outs.append(self._forward_one_samples(samples_))
            all_outs = torch.stack([a['pred_masks'] for a in all_outs], dim=0)
            mean_outs = all_outs.mean(0)
        return {'pred_masks': mean_outs}

    def _forward_one_samples(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # import ipdb; ipdb.set_trace()
        if self.backbone_name == 'resnet101' or self.backbone_name == 'resnet50':
            features, pos_list = self.backbone(samples)  # ## check warnings on floor_divide
            # import ipdb; ipdb.set_trace()
            T, _, H, W = features[0].tensors.shape
            batch_size = T // self.num_frames
        else:
            features = self.backbone(samples.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            batch_size, _, num_frames, _, _ = features[0].shape
            pos_list = []
            for i in range(len(features)):
                x = features[i].permute(0, 2, 1, 3, 4).flatten(0, 1)
                m = samples.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                x = NestedTensor(x, mask)
                features[i] = x
                pos_list.append(self.position_embedding(x).to(x.tensors.dtype))
        # import ipdb; ipdb.set_trace()
        if self.transformer is None:
            src, mask = features[-1].decompose()
            src_proj = self.input_proj(src)
            mask_ins = src_proj.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            # import ipdb; ipdb.set_trace()
            seg_feats = self.transformer(features, pos_list, batch_size)
            mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
        return out


class VOS_SwinMEDVTLPROP(VOS_SwinMEDVT):
    def __init__(self, args, backbone, backbone_dims, hidden_dim, transformer, num_frames,
                 pretrain_settings={}, lprop_mode=None, temporal_strides=[1], feat_loc=None,
                 stacked=1, n_class=80):
        super().__init__(args, backbone, backbone_dims, hidden_dim, transformer, num_frames, n_class=n_class)
        print('Initializing Swin-MEDVT-LPROP')
        self.stacked = stacked
        self.aux_loss = args.aux_loss > 0
        if feat_loc == 'early_coarse':
            feat_dim = 384
            hidden_dim = 128
        if feat_loc == 'early_fine':
            feat_dim = 384
            hidden_dim = 128
        elif feat_loc == 'late':
            feat_dim = 392
            hidden_dim = 128
        elif feat_loc == 'attmaps_only':
            feat_dim = 8
            hidden_dim = 16
        self.lprop_mode = lprop_mode
        # import ipdb;ipdb.set_trace()
        self.label_propagator = LabelPropagator(lprop_mode, feat_dim=feat_dim, hidden_dim=hidden_dim,
                                                label_scale=args.lprop_scale, n_class=n_class)
        self.feat_loc = feat_loc
        self.temporal_strides = temporal_strides

        # Pretraining Settings
        if pretrain_settings is None:
            pretrain_settings = {}

        if 'freeze_pretrained' not in pretrain_settings:
            pretrain_settings['freeze_pretrained'] = False
            pretrain_settings['pretrained_model_path'] = ''

        if 'pretrain_label_enc' not in pretrain_settings:
            pretrain_settings['pretrain_label_enc'] = False
        else:
            if pretrain_settings['pretrain_label_enc']:
                assert 'label_enc_pretrain_path' in pretrain_settings, \
                    "Label encoder pretrained weights path needed"

        if pretrain_settings['freeze_pretrained']:
            self._freeze_pretrained_modules()

        if pretrain_settings['pretrain_label_enc']:
            checkpoint = torch.load(pretrain_settings['label_enc_pretrain_path'])
            self.label_propagator.label_encoder.load_state_dict(checkpoint, strict=False)

    def _freeze_pretrained_modules(self):
        # import ipdb; ipdb.set_trace()
        # pretrained_modules = [self.vistr.backbone, self.vistr.transformer]
        """
        pretrained_modules = [self.backbone, self.transformer, self.insmask_head]
        for mod in pretrained_modules:
            for p in mod.parameters():
                p._requires_grad = False
                p.requires_grad = False
        """
        print('freezing pretrained ...')
        for name, p in self.named_parameters():
            if 'label_propagator' not in name:
                p._requires_grad = False
                p.requires_grad = False

    def _forward_one_samples(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # import ipdb; ipdb.set_trace()
        if self.backbone_name == 'resnet101' or self.backbone_name == 'resnet50':
            features, pos_list = self.backbone(samples)  # ## check warnings on floor_divide
        else:
            features = self.backbone(samples.tensors.permute(1, 0, 2, 3).unsqueeze(0))
            batch_size, _, num_frames, _, _ = features[0].shape
            pos_list = []
            for i in range(len(features)):
                x = features[i].permute(0, 2, 1, 3, 4).flatten(0, 1)
                m = samples.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                x = NestedTensor(x, mask)
                features[i] = x
                pos_list.append(self.position_embedding(x).to(x.tensors.dtype))
        # import ipdb; ipdb.set_trace()
        if self.transformer is None:
            src, mask = features[-1].decompose()
            src_proj = self.input_proj(src)
            mask_ins = src_proj.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            seg_feats = self.transformer(features, pos_list, batch_size)
            mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)

        # ##################################################################
        # import ipdb; ipdb.set_trace()
        outputs_seg_masks_b4lprop = self.insmask_head(mask_ins)
        outputs_seg_masks_b4lprop = outputs_seg_masks_b4lprop.squeeze(0)
        outputs_seg_masks_lprop = outputs_seg_masks_b4lprop.permute(1, 0, 2, 3)
        for i in range(self.stacked):
            if self.feat_loc == 'late':
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, seg_feats).squeeze(1)
            elif self.feat_loc == 'attmaps_only':
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, seg_feats[:, -8:]).squeeze(1)
            elif self.feat_loc == 'early_coarse':
                early_coarse_feats = features[-1].tensors
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, early_coarse_feats).squeeze(1)
            elif self.feat_loc == 'early_fine':
                early_fine_feats = features[0].tensors.squeeze(0)
                outputs_seg_masks_lprop = self.label_propagator(outputs_seg_masks_lprop, early_fine_feats).squeeze(1)
        # import ipdb; ipdb.set_trace()
        if self.lprop_mode in [1, 2]:
            if sum(outputs_seg_masks_lprop.shape[-2:]) < sum(outputs_seg_masks_b4lprop.shape[-2:]):
                outputs_seg_masks_lprop = F.interpolate(outputs_seg_masks_lprop, outputs_seg_masks_b4lprop.shape[-2:])
            elif sum(outputs_seg_masks_lprop.shape[-2:]) > sum(outputs_seg_masks_b4lprop.shape[-2:]):
                outputs_seg_masks_b4lprop = F.interpolate(outputs_seg_masks_b4lprop, outputs_seg_masks_lprop.shape[-2:])
            if self.training:
                # outputs_seg_masks = outputs_seg_masks_lprop
                outputs_seg_masks = torch.stack([outputs_seg_masks_b4lprop, outputs_seg_masks_lprop], dim=0).mean(0)
            else:
                outputs_seg_masks = torch.stack([outputs_seg_masks_b4lprop, outputs_seg_masks_lprop], dim=0).mean(0)
        elif self.lprop_mode == 3:
            outputs_seg_masks = outputs_seg_masks_lprop
        # import ipdb;ipdb.set_trace()
        out = {
            "pred_masks": outputs_seg_masks,
        }
        if self.aux_loss:
            out['aux_pred_masks'] = outputs_seg_masks_b4lprop
        # import ipdb;ipdb.set_trace()
        return out


def build_model_swin_medvt(args):
    print('using backbone:%s' % args.backbone)
    if args.backbone == 'swinS':
        backbone = build_swin_s_backbone(args.is_train, args.swin_s_pretrained_path)
        backbone_dims = (192, 384, 768, 768)
    elif args.backbone == 'swinB':
        backbone = build_swin_b_backbone(args.is_train, args.swin_b_pretrained_path)
        backbone_dims = (256, 512, 1024, 1024)
    else:
        raise ValueError('backbone: %s not implemented!' % args.backbone)
    # print('args.dim_feedforward:%d' % args.dim_feedforward)
    # import ipdb; ipdb.set_trace()
    transformer = Transformer(
        num_frames=args.num_frames,
        backbone_dims=backbone_dims,
        d_model=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=args.pre_norm,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_decoder_queries=args.num_queries,
        return_intermediate_dec=True,
        encoder_cross_layer=args.encoder_cross_layer
    )

    if args.is_train and transformer is not None:
        wcp_enc_upper_stages = False  # TODO check
        checkpoint = torch.load(args.resnet101_coco_weights_path, map_location='cpu')['model']
        # print('len(checkpoint): %d' % len(checkpoint))
        ckpt_keys = [k for k in checkpoint.keys()]
        del_keys_1 = [k for k in checkpoint.keys() if 'vistr.backbone.' in k]
        for kk in del_keys_1:
            del checkpoint[kk]
        # print('after removing backbone keys: len(checkpoint): %d' % len(checkpoint))
        ckpt_keys = [k for k in checkpoint.keys()]
        del_keys_2 = ['vistr.query_embed.weight', 'vistr.input_proj.weight', 'vistr.input_proj.bias']
        for kk in ckpt_keys:
            if 'vistr.class' in kk:
                del_keys_2.append(kk)
            if 'vistr.bbox' in kk:
                del_keys_2.append(kk)
            if 'mask_head.' in kk:
                # checkpoint[kk.replace('mask_head.', 'fpn.')] = checkpoint[kk]
                del_keys_2.append(kk)
            if 'vistr.transformer.' in kk:
                checkpoint[kk.replace('vistr.transformer.', '')] = checkpoint[kk]
                del_keys_2.append(kk)
        for kk in del_keys_2:
            del checkpoint[kk]
        # Copy decoder layer 6 weights to initialize next decoders
        if args.dec_layers > 6:  # and not args.finetune:
            cks = [k for k in checkpoint.keys() if 'decoder.layers.5.' in k]
            for i in range(6, args.dec_layers):
                for ck in cks:
                    mk = ck.replace('5', str(i))
                    checkpoint[mk] = checkpoint[ck].clone().detach()
        # import ipdb; ipdb.set_trace()
        enc_del_keys = []
        enc_add_weights = {}
        if sum(transformer.num_encoder_layers) > 0:
            for enc_stage in range(len(transformer.num_encoder_layers)):
                # import ipdb; ipdb.set_trace()
                if enc_stage == 0:
                    for lr_id in range(transformer.num_encoder_layers[0]):
                        if lr_id < 5:
                            cks = [k for k in checkpoint.keys() if 'encoder.layers.%d' % lr_id in k]
                            for ck in cks:
                                mk = ck.replace('encoder.layers.%d' % lr_id, 'encoder.layers.0.%d' % lr_id)
                                enc_add_weights[mk] = checkpoint[ck].clone().detach()
                                enc_del_keys.append(ck)
                        else:
                            cks = [k for k in checkpoint.keys() if 'encoder.layers.5' in k]
                            for ck in cks:
                                mk = ck.replace('encoder.layers.5', 'encoder.layers.0.%d' % lr_id)
                                enc_add_weights[mk] = checkpoint[ck].clone().detach()
                elif wcp_enc_upper_stages:
                    # import ipdb;ipdb.set_trace()
                    for lr_id in range(transformer.num_encoder_layers[enc_stage]):
                        cks = [k for k in checkpoint.keys() if 'encoder.layers.%d' % lr_id in k]
                        for ck in cks:
                            if 'norm' in ck:
                                continue
                            mk = ck.replace('encoder.layers.%d' % lr_id, 'encoder.layers.%d.%d' % (enc_stage, lr_id))
                            enc_add_weights[mk] = checkpoint[ck].clone().detach()
        if args.encoder_cross_layer and False:
            cks = [k for k in checkpoint.keys() if 'encoder.layers.0.' in k]
            for ck in cks:
                if 'norm' in ck:
                    continue
                mk = ck.replace('layers.0.', 'cross_res_layers.0.')
                enc_add_weights[mk] = checkpoint[ck].clone().detach()
                # mk = ck.replace('layers.0.', 'cross_res_layers.1.')
        # import ipdb; ipdb.set_trace()
        for dk in enc_del_keys:
            del checkpoint[dk]
        for kk, vv in enc_add_weights.items():
            checkpoint[kk] = vv
        # print('len(state_dict): %d' % len(checkpoint))
        # print('len(transformer.state_dict()): %d' % len(transformer.state_dict()))
        matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys()]
        # print('matched keys:%d' % len(matched_keys))
        # import ipdb;ipdb.set_trace()
        shape_mismatch = []
        # import re
        use_partial_match = True
        for kid, kk in enumerate(matched_keys):
            # if kid>66:
            # import ipdb;ipdb.set_trace()
            # print('kid:%d kk:%s'%(kid, kk))
            if checkpoint[kk].shape != transformer.state_dict()[kk].shape:
                # print('shape not matched key:%s'%kk)
                # TODO check with partial copy
                # import ipdb;ipdb.set_trace()
                if not use_partial_match:
                    shape_mismatch.append(kk)
                    continue
                if 'encoder.' in kk:
                    if 'linear1.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward, :].clone().detach()
                    elif 'linear1.bias' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                    elif 'linear2.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                    else:
                        import ipdb;
                        ipdb.set_trace()
                        shape_mismatch.append(kk)
                elif 'decoder.' in kk:
                    if 'linear1.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward, :].clone().detach()
                    elif 'linear1.bias' in kk:
                        checkpoint[kk] = checkpoint[kk][:args.dim_feedforward].clone().detach()
                    elif 'linear2.weight' in kk:
                        checkpoint[kk] = checkpoint[kk][:, :args.dim_feedforward].clone().detach()
                    else:
                        import ipdb;
                        ipdb.set_trace()
                        shape_mismatch.append(kk)
                else:
                    import ipdb;
                    ipdb.set_trace()
                    shape_mismatch.append(kk)
                    # print('here')
        # print('len(shape_mismatch):%d' % len(shape_mismatch))
        for kk in shape_mismatch:
            del checkpoint[kk]
        transformer.load_state_dict(checkpoint, strict=False)
        # shape_matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys()]
        # print('shape_matched keys:%d' % len(shape_matched_keys))
    if args.lprop_mode > 0:
        temporal_strides = [1] if not hasattr(args, 'temporal_strides') else args.temporal_strides
        model = VOS_SwinMEDVTLPROP(args, backbone, backbone_dims=backbone_dims, hidden_dim=args.hidden_dim,
                                   transformer=transformer, num_frames=args.num_frames,
                                   pretrain_settings=args.pretrain_settings,
                                   lprop_mode=args.lprop_mode, temporal_strides=temporal_strides,
                                   feat_loc=args.feat_loc, stacked=args.stacked_lprop, n_class=args.num_classes)
    else:
        model = VOS_SwinMEDVT(args, backbone, backbone_dims=backbone_dims, hidden_dim=args.hidden_dim,
                              transformer=transformer, num_frames=args.num_frames, n_class=args.num_classes)

    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    if args.is_train:
        criterion = criterions.SetMultiLabelCriterion(weight_dict=weight_dict, losses=losses,
                                                      aux_loss=args.aux_loss, aux_loss_norm=args.aux_loss_norm)
    else:
        criterion = criterions.SetMultiLabelCriterion(weight_dict=weight_dict, losses=losses)
    criterion.to(torch.device(args.device))
    return model, criterion
