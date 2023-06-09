"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
import torchvision.ops
import copy
from typing import Optional, List, OrderedDict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import math
import logging

from avos.models.medvt_swin import build_swin_b_backbone
from avos.models.medvt_swin import build_swin_s_backbone
from avos.models.medvt_swin import MaskHeadSmallConv, MHAttentionMap
from avos.models.position_encoding import build_position_encoding
from avos.models.utils import get_clones, get_activation_fn
from actor_action.models import criterions
from actor_action.models.label_propagation import LabelPropagator as LabelPropagator
from actor_action.util.misc import (nested_tensor_from_tensor_list)
from actor_action.util.misc import NestedTensor, is_main_process

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Transformer(nn.Module):

    def __init__(self, num_frames, backbone_dims, d_model=384, nhead=8,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 num_encoder_layers=(6,),
                 num_decoder_layers=6,
                 num_decoder_queries=1,
                 return_intermediate_dec=False,
                 bbox_nhead=8,
                 encoder_cross_layer=False,
                 decoder_multiscale=True):
        """
        Args:
            num_frames:
            backbone_dims:
            d_model:
            nhead:
            dim_feedforward:
            dropout:
            activation:
            normalize_before:
            num_encoder_layers:
            num_decoder_layers:
            num_decoder_queries:
            return_intermediate_dec:
            bbox_nhead:
            encoder_cross_layer:
            decoder_multiscale:
        """
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.tgt = None  # TODO Check
        if not type(num_encoder_layers) in [list, tuple]:
            num_encoder_layers = [num_encoder_layers]
        self.num_encoder_layers = [num_encoder_layer if type(num_encoder_layer) is int else int(num_encoder_layer) for
                                   num_encoder_layer in num_encoder_layers]
        self.num_decoder_queries = num_decoder_queries
        self.decoder_multiscale = decoder_multiscale
        self.backbone_dims = backbone_dims
        self.num_backbone_feats = len(backbone_dims)
        self.num_frames = num_frames
        self.input_proj_modules = nn.ModuleList()
        self.use_encoder = sum(self.num_encoder_layers) > 0
        self.num_encoder_stages = len(self.num_encoder_layers)
        self.use_decoder = num_decoder_layers > 0
        self.bbox_nhead = bbox_nhead

        if self.num_encoder_stages == 1 and encoder_cross_layer:
            self.num_encoder_stages = 2

        for backbone_dim in backbone_dims:
            self.input_proj_modules.append(nn.Conv2d(backbone_dim, d_model, kernel_size=1))
        if sum(self.num_encoder_layers) > 0:
            self.encoder = TransformerEncoder(self.num_encoder_layers, nhead, dim_feedforward, d_model, dropout,
                                              activation,
                                              normalize_before, use_cross_layers=encoder_cross_layer,
                                              cross_pos='cascade')

        if num_decoder_layers > 0:
            self.query_embed = nn.Embedding(num_decoder_queries, d_model)
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)
            self.bbox_attention = MHAttentionMap(d_model, d_model, bbox_nhead, dropout=0.0)

        if num_decoder_layers > 0 and not self.decoder_multiscale:
            self.fpn = MaskHeadSmallConv(dim=d_model + 8, context_dim=d_model)
        else:
            self.fpn = MaskHeadSmallConv(dim=d_model, context_dim=d_model)

        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, pos_list, batch_size):
        # features as TCHW
        # import ipdb; ipdb.set_trace()
        bt = features[-1].tensors.shape[0]
        bs_f = bt // self.num_frames
        # project all backbone features to transformer dim
        for i in range(len(features)):
            src, mask = features[i].decompose()
            assert mask is not None
            src_proj = self.input_proj_modules[i](src)
            features[i] = NestedTensor(src_proj, mask)

        # reshape all features to sequences for encoder
        if self.use_encoder:
            enc_feat_list = []
            enc_mask_list = []
            enc_pos_embd_list = []
            enc_feat_shapes = []
            pred_mems_flat = []
            for i in range(self.num_encoder_stages):
                feat_index = -1 - i  # taking in reverse order from deeper to shallow
                src_proj, mask = features[feat_index].decompose()
                assert mask is not None
                n, c, s_h, s_w = src_proj.shape
                enc_feat_shapes.append((s_h, s_w))
                src = src_proj.reshape(bs_f, self.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
                # bs, c, t, hw = src.shape
                src = src.flatten(2).permute(2, 0, 1)
                enc_feat_list.append(src)
                mask = mask.reshape(bs_f, self.num_frames, s_h * s_w)
                mask = mask.flatten(1)
                enc_mask_list.append(mask)
                pos_embed = pos_list[feat_index].permute(0, 2, 1, 3, 4).flatten(-2)
                pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                enc_pos_embd_list.append(pos_embed)
            encoder_features = self.encoder(features=enc_feat_list, src_key_padding_masks=enc_mask_list,
                                            pos_embeds=enc_pos_embd_list, sizes=enc_feat_shapes,
                                            pred_mems=pred_mems_flat)
            for i in range(self.num_encoder_stages):
                memory_i = encoder_features[i]
                h, w = enc_feat_shapes[i]
                memory_i = memory_i.permute(1, 2, 0).view(bs_f, self.d_model, self.num_frames, h * w)
                memory_i = memory_i.permute(0, 2, 1, 3).reshape(bs_f, self.num_frames, self.d_model, h, w).flatten(0, 1)
                encoder_features[i] = memory_i
                # print('enc output>> i:%d  memory_i.shape:%s' % (i, memory_i.shape))
            deep_feature = encoder_features[0]
            fpn_features = []
            for i in range(1, self.num_encoder_stages):
                fpn_features.append(encoder_features[i])
            for i in reversed(range(self.num_backbone_feats - self.num_encoder_stages)):
                _, c_f, h, w = features[i].tensors.shape
                features[i].tensors = features[i].tensors.reshape(bs_f, self.num_frames, c_f, h, w)
                fpn_features.append(features[i].tensors.flatten(0, 1))
        else:
            # print('Not using encoder>>> Not implemented yet as not doing experiment now')
            # import ipdb;ipdb.set_trace()
            # TODO check, not tested
            _, c_f, h, w = features[-1].tensors.shape
            features[-1].tensors = features[-1].tensors.reshape(bs_f, self.num_frames, c_f, h, w)
            deep_feature = features[-1].tensors.flatten(0, 1)
            fpn_features = []
            for i in reversed(range(self.num_backbone_feats - 1)):
                _, c_f, h, w = features[i].tensors.shape
                features[i].tensors = features[i].tensors.reshape(bs_f, self.num_frames, c_f, h, w)
                fpn_features.append(features[i].tensors.flatten(0, 1))
        ################################################################
        ###################################################################
        # import ipdb; ipdb.set_trace()
        if self.use_decoder:
            if self.decoder_multiscale:
                ms_feats = self.fpn(deep_feature, fpn_features)
                hr_feat = ms_feats[-1]

                dec_features = []
                pos_embed_list = []
                size_list = []
                dec_mask_list = []
                for i in range(3):
                    fi = ms_feats[i]
                    ni, ci, hi, wi = fi.shape
                    fi = fi.reshape(bs_f, self.num_frames, ci, hi, wi).permute(0, 2, 1, 3, 4).flatten(-2).flatten(
                        2).permute(2, 0,
                                   1)
                    dec_mask_i = features[-1 - i].mask.reshape(bs_f, self.num_frames, hi * wi).flatten(1)
                    pe = pos_list[-1 - i].permute(0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)
                    dec_features.append(fi)
                    pos_embed_list.append(pe)
                    size_list.append((hi, wi))
                    dec_mask_list.append(dec_mask_i)
                    prev_mem_4_dec = None

                query_embed = self.query_embed.weight
                query_embed = query_embed.unsqueeze(1)
                tq, bq, cq = query_embed.shape
                query_embed = query_embed.repeat(self.num_frames // tq, bs_f, 1)

                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, dec_features, memory_key_padding_mask=dec_mask_list,
                                  pos=pos_embed_list, query_pos=query_embed, size_list=size_list,
                                  prev_pred_mem=prev_mem_4_dec)
                hs = hs.transpose(1, 2)
                n_f = 1 if self.num_decoder_queries == 1 else self.num_decoder_queries // self.num_frames
                obj_attn_masks = []
                for i in range(self.num_frames):
                    t2, c2, h2, w2 = hr_feat.shape
                    hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
                    memory_f = hr_feat[i, :, :, :].reshape(batch_size, c2, h2, w2)
                    mask_f = features[0].mask[i, :, :].reshape(batch_size, h2, w2)
                    obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
                    obj_attn_masks.append(obj_attn_mask_f)
                obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
                seg_feats = torch.cat([hr_feat, obj_attn_masks], dim=1)

            else:
                ####################################
                dec_features = []
                pos_embed_list = []
                size_list = []
                dec_mask_list = []
                # import ipdb; ipdb.set_trace()
                fi = deep_feature
                ni, ci, hi, wi = fi.shape
                fi = fi.reshape(bs_f, self.num_frames, ci, hi, wi).permute(0, 2, 1, 3, 4).flatten(-2).flatten(
                    2).permute(2, 0, 1)
                dec_mask_i = features[-1].mask.reshape(bs_f, self.num_frames, hi * wi).flatten(1)
                pe = pos_list[-1].permute(0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)
                dec_features.append(fi)
                pos_embed_list.append(pe)
                size_list.append((hi, wi))
                dec_mask_list.append(dec_mask_i)

                query_embed = self.query_embed.weight
                query_embed = query_embed.unsqueeze(1)
                tq, bq, cq = query_embed.shape
                query_embed = query_embed.repeat(self.num_frames // tq, bs_f, 1)

                # import ipdb; ipdb.set_trace()
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, dec_features, memory_key_padding_mask=dec_mask_list,
                                  pos=pos_embed_list, query_pos=query_embed, size_list=size_list)
                hs = hs.transpose(1, 2)
                n_f = 1 if self.num_decoder_queries == 1 else self.num_decoder_queries // self.num_frames
                obj_attn_masks = []
                # import ipdb; ipdb.set_trace()
                for i in range(self.num_frames):
                    ni, ci, hi, wi = deep_feature.shape
                    hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
                    memory_f = deep_feature[i, :, :, :].reshape(batch_size, ci, hi, wi)
                    mask_f = features[-1].mask[i, :, :].reshape(batch_size, hi, wi)
                    obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
                    obj_attn_masks.append(obj_attn_mask_f)
                # import ipdb;ipdb.set_trace()
                obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
                deep_feature = torch.cat([deep_feature, obj_attn_masks], dim=1)
                ##################################
                ms_feats = self.fpn(deep_feature, fpn_features)
                hr_feat = ms_feats[-1]
                seg_feats = hr_feat
        else:
            ms_feats = self.fpn(deep_feature, fpn_features)
            hr_feat = ms_feats[-1]
            seg_feats = hr_feat
        return seg_feats


class TransformerEncoder(nn.Module):

    def __init__(self, num_encoder_layers, nhead, dim_feedforward, d_model, dropout, activation, normalize_before,
                 use_cross_layers, cross_pos):
        super().__init__()
        # ######################################
        # import ipdb; ipdb.set_trace()
        self.cross_pos = cross_pos  # pre, post, cascade
        self.num_layers = num_encoder_layers
        self.num_stages = len(num_encoder_layers)
        self.layers = nn.ModuleList()

        for i in range(len(num_encoder_layers)):
            # print('Encoder stage:%d dim_feedforward:%d' % (i, dim_feedforward))
            self.layers.append(nn.ModuleList())
            for j in range(num_encoder_layers[i]):
                _nhead = nhead if i == 0 else 1
                _dim_feedforward = dim_feedforward if i == 0 else d_model
                _dropout = dropout if i == 0 else 0
                _use_layer_norm = (i == 0)
                encoder_layer = TransformerEncoderLayer(d_model, _nhead, _dim_feedforward, _dropout, activation,
                                                        normalize_before, _use_layer_norm)
                self.layers[i].append(encoder_layer)
        # ######################################
        # import ipdb; ipdb.set_trace()
        # #############################################
        self.norm = nn.LayerNorm(d_model) if normalize_before else None
        self.cross_pos = cross_pos
        if self.num_stages > 1:
            self.norms = None if not normalize_before else nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(self.num_stages - 1)])
        self.use_cross_layers = use_cross_layers
        if use_cross_layers:
            self.cross_res_layers = nn.ModuleList([
                CrossResolutionEncoderLayer(d_model=384, nhead=1, dim_feedforward=384, layer_norm=False,
                                            custom_instance_norm=True, fuse='linear')
            ])

    @staticmethod
    def reshape_up_reshape(src0, src0_shape, src1_shape):
        # import ipdb; ipdb.set_trace()
        s_h, s_w = src0_shape
        s_h2, s_w2 = src1_shape
        t_f = src0.shape[0] // (s_h * s_w)
        src0_up = src0.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
        src0_up = src0_up.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
        src0_up = F.interpolate(src0_up, (s_h2, s_w2), mode='bilinear')
        n2, c2, s_h2, s_w2 = src0_up.shape
        src0_up = src0_up.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
        src0_up = src0_up.flatten(2).permute(2, 0, 1)
        return src0_up

    def forward(self, features, src_key_padding_masks, pos_embeds, sizes, masks=None, pred_mems=None):
        """
        Args:
            features:
            src_key_padding_masks:
            pos_embeds:
            sizes: [(h_i,w_i),...]
            masks:
            pred_mems:

        Returns:

        """
        outputs = []
        # import ipdb; ipdb.set_trace()
        for i in range(self.num_stages):
            output = features[i]
            skp_mask = src_key_padding_masks[i]
            pos_embed = pos_embeds[i]
            pred_mem_i = None
            if pred_mems is not None and len(pred_mems) > 0:
                pred_mem_i = pred_mems[i]
            if self.use_cross_layers and i > 0 and self.cross_pos == 'cascade':
                src0 = outputs[i - 1]
                src1 = output
                mask0 = src_key_padding_masks[i - 1]
                pos0 = pos_embeds[i - 1]
                mask1 = src_key_padding_masks[i]
                pos1 = pos_embeds[i]
                # ############################################################
                # Resize smaller one to match shape ##########################
                # ############################################################
                if src0.shape[0] != src1.shape[0]:
                    # import ipdb; ipdb.set_trace()
                    src0_up = self.reshape_up_reshape(src0, src0_shape=sizes[i - 1], src1_shape=sizes[i])
                    pos0_up = pos1
                    mask0_up = mask1
                else:
                    src0_up = src0
                    pos0_up = pos0
                    mask0_up = mask0
                # ############################################################
                output = self.cross_res_layers[0](src0_up, src1, mask0_up, mask1, pos0_up, pos1)
            # shape = sizes[i]
            # import ipdb; ipdb.set_trace()
            for j in range(self.num_layers[i]):
                output = self.layers[i][j](output, src_mask=None, src_key_padding_mask=skp_mask, pos=pos_embed,
                                           pred_mem=pred_mem_i)
            outputs.append(output)
        if self.use_cross_layers and self.cross_pos == 'post':
            src0 = outputs[0]
            if self.num_stages > 1:
                src1 = outputs[1]
            else:
                src1 = features[1]
                outputs.append(src1)
            mask0 = src_key_padding_masks[0]
            pos0 = pos_embeds[0]
            mask1 = src_key_padding_masks[1]
            pos1 = pos_embeds[1]
            # ############################################################
            # Resize smaller one to match shape ##########################
            # ############################################################
            if src0.shape[0] != src1.shape[0]:
                # import ipdb; ipdb.set_trace()
                src0_up = self.reshape_up_reshape(src0, src0_shape=sizes[i - 1], src1_shape=sizes[i])
                pos0_up = pos1
                mask0_up = mask1
            else:
                src0_up = src0
                pos0_up = pos0
                mask0_up = mask0
            # ############################################################
            output1 = self.cross_res_layers[0](src0_up, src1, mask0_up, mask1, pos0_up, pos1)
            # import ipdb;ipdb.set_trace()
            outputs[1] = output1
        # TODO
        # Check this norm later
        if self.norm is not None:
            outputs[0] = self.norm(outputs[0])
            if self.num_stages > 1:
                for i in range(self.num_stages - 1):
                    outputs[i + 1] = self.norms[i](outputs[i + 1])
        return outputs


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, dec_features,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                size_list=None, **kwargs):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            feat_index = idx % (len(dec_features))
            current_dec_feat = dec_features[feat_index]
            pos_i = pos[feat_index]
            memory_key_padding_mask_i = memory_key_padding_mask[feat_index]
            output = layer(output, current_dec_feat, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask_i,
                           pos=pos_i, query_pos=query_pos, size_list=size_list)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class CrossResolutionEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, layer_norm, custom_instance_norm, fuse,
                 dropout=0, activation="relu"):
        super().__init__()
        print('Creating cross resolution layer>>> d_model: %3d nhead:%d dim_feedforward:%3d' % (
            d_model, nhead, dim_feedforward))
        assert fuse is None or fuse in ['simple', 'linear', 'bilinear']
        self.fuse = fuse  # None, 'simple, ''', 'linear', 'bilinear'
        self.nhead = nhead
        if self.fuse is None:
            pass
        elif self.fuse == 'simple':
            pass
        elif self.fuse == 'linear':
            self.fuse_linear1 = nn.Linear(d_model, d_model)
            self.fuse_linear2 = nn.Linear(d_model, d_model)
            self.fuse_linear3 = nn.Linear(d_model, d_model)
        elif self.fuse == 'bilinear':
            self.bilinear = nn.Bilinear(d_model, d_model, d_model)
        else:
            raise ValueError('fuse:{} not recognized'.format(self.fuse))
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.custom_instance_norm = custom_instance_norm
        self.norm1 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.res_embed = nn.Embedding(1, d_model)
        self.res_embed_shallow = nn.Embedding(1, d_model)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_deep, feat_shallow, mask_deep, mask_shallow, pos_deep, pos_shallow):
        # import ipdb; ipdb.set_trace()
        if self.fuse is None:
            _f_deep = feat_deep
        elif self.fuse == 'simple':
            _f_deep = feat_deep + feat_shallow
        elif self.fuse == 'linear':
            _f_deep = self.fuse_linear3(
                self.activation(self.fuse_linear1(feat_deep)) + self.activation(self.fuse_linear2(feat_shallow)))
        elif self.fuse == 'bilinear':
            _f_deep = self.bilinear(feat_deep, feat_shallow)
        else:
            raise ValueError('fuse:{} not recognized'.format(self.fuse))
        res_embed = self.res_embed.weight.unsqueeze(0).repeat(pos_deep.shape[0], 1, 1)
        res_embed_shallow = self.res_embed_shallow.weight.unsqueeze(0).repeat(pos_shallow.shape[0], 1, 1)
        if self.custom_instance_norm:
            deep_u = feat_deep.mean(dim=0)
            deep_s = feat_deep.std(dim=0)
            shallow_u = feat_shallow.mean(dim=0)
            shallow_s = feat_shallow.std(dim=0)
            _f_deep = (_f_deep - deep_u)
            if deep_s.min() > 1e-10:
                _f_deep = _f_deep / deep_s
            _f_deep = (_f_deep * shallow_s) + shallow_u
        _f_deep = _f_deep * self.gamma + self.beta
        kp = _f_deep + pos_deep + res_embed
        qp = feat_shallow + pos_shallow + res_embed_shallow
        vv = feat_shallow
        # import ipdb; ipdb.set_trace()
        attn_mask = torch.mm(mask_shallow.transpose(1, 0).double(), mask_deep.double()).bool()
        out = self.self_attn(qp, kp, value=vv, attn_mask=attn_mask, key_padding_mask=mask_shallow)[0]
        out = feat_shallow + self.dropout1(out)
        out = self.norm1(out)
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm2(out)
        return out


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_layer_norm=True):
        super().__init__()
        self.nhead = nhead
        self.use_layer_norm = use_layer_norm
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     pred_mem: Tensor = None):
        # import ipdb; ipdb.set_trace()
        q = k = self.with_pos_embed(src, pos)
        self_attn_mask = src_mask
        if pred_mem is not None:
            # #########
            th = 0.2
            pred_mem_mask = pred_mem <= th
            if src_key_padding_mask is not None:
                src_key_padding_mask = torch.logical_or(pred_mem_mask, src_key_padding_mask).detach()
            else:
                src_key_padding_mask = pred_mem_mask

        src2 = self.self_attn(q, k, value=src, attn_mask=self_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # import ipdb; ipdb.set_trace()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # import ipdb; ipdb.sset_trace()
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pred_mem: Tensor = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, pred_mem=pred_mem)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     size_list=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        query_ca = self.with_pos_embed(tgt, query_pos)
        key_ca = self.with_pos_embed(memory, pos)
        value_ca = memory
        tgt2 = self.multihead_attn(query=query_ca,
                                   key=key_ca,
                                   value=value_ca, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                size_list=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, size_list=size_list)


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
