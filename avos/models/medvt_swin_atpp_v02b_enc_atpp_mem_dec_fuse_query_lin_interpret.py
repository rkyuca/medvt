"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
import torchvision.ops
from typing import Optional, List, OrderedDict, Dict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
import logging

from avos.models.utils import get_clones, get_activation_fn, expand
from avos.models import criterions
from avos.models.swin_transformer_3d import SwinTransformer3D
from avos.utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from avos.models.label_propagation import LabelPropagator
from avos.models.position_encoding import build_position_encoding

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, context_dim):
        super().__init__()
        # print('Creating FPN-> dim:%d context_dim:%d' % (dim, context_dim))
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.conv_offset = torch.nn.Conv2d(inter_dims[3], 18, 1)  # , bias=False)
        self.dcn = torchvision.ops.DeformConv2d(inter_dims[3], inter_dims[4], 3, padding=1, bias=False)
        self.dim = dim

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, fpns: List[Tensor]):
        multi_scale_features = []
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = fpns[0]
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = fpns[1]
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = fpns[2]
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # dcn for the last layer
        offset = self.conv_offset(x)
        x = self.dcn(x, offset)
        x = self.gn5(x)
        x = F.relu(x)
        multi_scale_features.append(x)
        return multi_scale_features


class MHAttentionMapTube(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = 1.0  # float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        """
        Args:
            q: 1x n_head x c
            k: T x C x H x W
            mask:
        Returns:
        """
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        # import ipdb; ipdb.set_trace()
        weights = torch.einsum('bqc,nchw->bnqhw', q, k)
        # import ipdb; ipdb.set_trace()
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(0), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


class EncATPP(nn.Module):
    def __init__(self, input_dim=384, model_dim=384, output_dim=384, num_branches=1, groups=4, dropout=0.1,
                 use_res=True):
        super(EncATPP, self).__init__()
        self.use_res = use_res
        self.branches = nn.ModuleList()
        # import ipdb; ipdb.set_trace()
        for ii in range(1, num_branches + 1):
            branch = nn.Sequential(
                nn.Conv3d(in_channels=input_dim, out_channels=model_dim, kernel_size=(3, 3, 3), padding='same',
                          dilation=(ii + 1, 1, 1)),
                nn.GroupNorm(groups, model_dim),
                nn.ReLU())
            self.branches.add_module('branch_%d' % ii, branch)
        if self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=(1, 1, 1), padding='same',
                          dilation=(1, 1, 1)),
                nn.GroupNorm(groups, output_dim),
                nn.ReLU())

        self.out_proj = nn.Sequential(
            nn.Conv3d(in_channels=model_dim * num_branches, out_channels=output_dim, kernel_size=(1, 1, 1),
                      padding='same', dilation=(1, 1, 1)),
            nn.GroupNorm(groups, output_dim),
            nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        # self._init_weight()

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        branch_outs = []
        for branch in self.branches:
            bo = branch(x)
            branch_outs.append(bo)
        bo = torch.cat(branch_outs, dim=1)
        if self.use_res:
            out = self.shortcut(x) + self.out_proj(self.dropout(bo))
        else:
            out = self.out_proj(self.dropout(bo))
        return out


class DecATPP(nn.Module):
    def __init__(self, input_dim=384, model_dim=384, output_dim=384, num_branches=1, groups=4, dropout=0.1,
                 use_res=True):
        super(DecATPP, self).__init__()
        self.use_res = use_res
        self.branches = nn.ModuleList()
        for ii in range(1, num_branches + 1):
            branch = nn.Sequential(
                nn.Conv3d(in_channels=input_dim, out_channels=model_dim, kernel_size=(3, 3, 3), padding='same',
                          dilation=(ii, 1, 1)),
                nn.GroupNorm(groups, model_dim),
                nn.ReLU())
            self.branches.add_module('branch_%d' % ii, branch)
        if self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=(1, 1, 1), padding='same',
                          dilation=(1, 1, 1)),
                nn.GroupNorm(groups, output_dim),
                nn.ReLU())

        self.out_proj = nn.Sequential(
            nn.Conv3d(in_channels=model_dim * num_branches, out_channels=output_dim, kernel_size=(1, 1, 1),
                      padding='same', dilation=(1, 1, 1)),
            nn.GroupNorm(groups, output_dim),
            nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        # self._init_weight()

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        branch_outs = []
        for branch in self.branches:
            bo = branch(x)
            branch_outs.append(bo)
        bo = torch.cat(branch_outs, dim=1)
        # import ipdb;ipdb.set_trace()
        if self.use_res:
            out = self.shortcut(x) + self.out_proj(self.dropout(bo))
        else:
            out = self.out_proj(self.dropout(bo))
        return out


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
                 decoder_multiscale=True,
                 decoder_attn_fuse='cat',
                 decoder_bbox_head='tube',
                 num_classes=1,
                 enc_atpp_branch=1,
                 enc_atpp_res=False,
                 dec_atpp_branch=1,
                 dec_atpp_res=False,
                 dec_feat_atpp_branches=1,
                 dec_feat_atpp_res=1,
                 ):
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
            decoder_attn_fuse: 'cat', one of 'add'/'cat'
            decoder_bbox_head: 'tube','frame'
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
        self.decoder_attn_fuse = decoder_attn_fuse
        self.bbox_nhead = bbox_nhead
        self.decoder_bbox_head = decoder_bbox_head
        self.num_classes = num_classes

        self.dec_atpp_branch = dec_atpp_branch
        self.dec_atpp_res = dec_atpp_res
        self.dec_feat_atpp = dec_feat_atpp_branches > 0
        self.dec_feat_atpp_branches = dec_feat_atpp_branches
        self.dec_feat_atpp_res = dec_feat_atpp_res

        if self.num_encoder_stages == 1 and encoder_cross_layer:
            self.num_encoder_stages = 2

        # import ipdb;ipdb.set_trace()
        for backbone_dim in backbone_dims:
            self.input_proj_modules.append(nn.Conv2d(backbone_dim, d_model, kernel_size=1))
        if sum(self.num_encoder_layers) > 0:
            self.encoder = TransformerEncoder(self.num_encoder_layers, nhead, dim_feedforward, d_model, dropout,
                                              activation,
                                              normalize_before, use_cross_layers=encoder_cross_layer,
                                              cross_pos='cascade', num_classes=self.num_classes,
                                              enc_atpp_branch=enc_atpp_branch, enc_atpp_res=enc_atpp_res)

        if num_decoder_layers > 0:
            if decoder_bbox_head == 'tube':
                self.query_embed = nn.Embedding(bbox_nhead, d_model)
                self.bbox_attention = MHAttentionMapTube(d_model, d_model, bbox_nhead, dropout=0.0)
            else:
                self.query_embed = nn.Embedding(num_decoder_queries, d_model)
                self.bbox_attention = MHAttentionMap(d_model, d_model, bbox_nhead, dropout=0.0)

            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,
                                                    dec_atpp_branch=self.dec_atpp_branch,
                                                    dec_atpp_res=self.dec_atpp_res)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

            if self.decoder_attn_fuse is not None and self.decoder_attn_fuse == 'add':
                self.bbox_sal_adapter = nn.Sequential(nn.Conv2d(bbox_nhead, d_model, kernel_size=1))

        if num_decoder_layers > 0 and not self.decoder_multiscale:
            self.fpn = MaskHeadSmallConv(dim=d_model + 8, context_dim=d_model)
        else:
            self.fpn = MaskHeadSmallConv(dim=d_model, context_dim=d_model)
        if self.dec_feat_atpp:
            self.dec_atpp = DecATPP(input_dim=392, model_dim=392, output_dim=392, num_branches=dec_feat_atpp_branches,
                                    use_res=dec_feat_atpp_res)
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

        trans_stage_outputs = []

        bt = features[-1].tensors.shape[0]
        bs_f = bt // self.num_frames
        # project all backbone features to transformer dim
        # import ipdb;ipdb.set_trace()
        for i in range(len(features)):
            src, mask = features[i].decompose()
            assert mask is not None
            src_proj = self.input_proj_modules[i](src)
            features[i] = NestedTensor(src_proj, mask)
            trans_stage_outputs.append(src_proj.clone())  # TxCxHxW

        # reshape all features to sequences for encoder
        if self.use_encoder:
            enc_feat_list = []
            enc_mask_list = []
            enc_pos_embd_list = []
            enc_feat_shapes = []
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
                # import ipdb;ipdb.set_trace()
                # print('inside enc_features>> i:%d feat_index:%d src.shape:%s mask.shape:%s pos.shape:%s' % (
                # i, feat_index, str(src.shape), str(mask.shape), str(pos_embed.shape)))
            encoder_features, enc_intermediates = self.encoder(features=enc_feat_list,
                                                               src_key_padding_masks=enc_mask_list,
                                                               pos_embeds=enc_pos_embd_list, sizes=enc_feat_shapes)
            trans_stage_outputs = trans_stage_outputs + enc_intermediates
            for i in range(self.num_encoder_stages):
                memory_i = encoder_features[i]
                h, w = enc_feat_shapes[i]
                memory_i = memory_i.permute(1, 2, 0).view(bs_f, self.d_model, self.num_frames, h * w)
                memory_i = memory_i.permute(0, 2, 1, 3).reshape(bs_f, self.num_frames, self.d_model, h, w).flatten(0, 1)
                encoder_features[i] = memory_i
                # print('enc output>> i:%d  memory_i.shape:%s' % (i, memory_i.shape))

            # for enc_feat in encoder_features:
            #    trans_stage_outputs.append(enc_feat.permute(1, 0, 2, 3).flatten(1).mean(1).unsqueeze(0))

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
                if self.decoder_bbox_head == 'tube':
                    query_embed = self.query_embed.weight
                    query_embed = query_embed.unsqueeze(1)
                else:
                    query_embed = self.query_embed.weight
                    query_embed = query_embed.unsqueeze(1)
                    tq, bq, cq = query_embed.shape
                    query_embed = query_embed.repeat(self.num_frames // tq, bs_f, 1)
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, dec_features, memory_key_padding_mask=dec_mask_list,
                                  pos=pos_embed_list, query_pos=query_embed, size_list=size_list)
                hs = hs.transpose(1, 2)
                dec_intermediates = [hs[i].squeeze(0).unsqueeze(2).unsqueeze(3) for i in range(len(hs))]
                trans_stage_outputs = trans_stage_outputs + dec_intermediates
                if self.decoder_bbox_head == 'tube':
                    obj_attn_masks = self.bbox_attention(hs[-1], hr_feat, mask=features[0].mask).flatten(0, 1)
                else:
                    n_f = 1 if self.num_decoder_queries == 1 else self.num_decoder_queries // self.num_frames
                    obj_attn_masks = []
                    # import ipdb; ipdb.set_trace()
                    for i in range(self.num_frames):
                        t2, c2, h2, w2 = hr_feat.shape
                        hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
                        memory_f = hr_feat[i, :, :, :].reshape(batch_size, c2, h2, w2)
                        mask_f = features[0].mask[i, :, :].reshape(batch_size, h2, w2)
                        obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
                        obj_attn_masks.append(obj_attn_mask_f)
                    obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
                # import ipdb;ipdb.set_trace()
                if self.decoder_attn_fuse is not None and self.decoder_attn_fuse == 'add':
                    obj_attn_masks = self.bbox_sal_adapter(obj_attn_masks)
                    seg_feats = hr_feat + obj_attn_masks
                else:
                    seg_feats = torch.cat([hr_feat, obj_attn_masks], dim=1)
                if self.dec_feat_atpp:
                    seg_feats = self.dec_atpp(seg_feats.permute(1, 0, 2, 3).unsqueeze(0)).squeeze(0).permute(1, 0, 2, 3)
                # trans_stage_outputs.append(seg_feats.permute(1, 0, 2, 3).flatten(1).mean(1).unsqueeze(0))
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
                # import ipdb;
                # ipdb.set_trace()
                if self.decoder_attn_fuse is not None and self.decoder_attn_fuse == 'add':
                    obj_attn_masks = self.bbox_sal_adapter(obj_attn_masks)
                    deep_feature = deep_feature + obj_attn_masks
                else:
                    deep_feature = torch.cat([deep_feature, obj_attn_masks], dim=1)
                ##################################
                ms_feats = self.fpn(deep_feature, fpn_features)
                hr_feat = ms_feats[-1]
                seg_feats = hr_feat
        else:
            ms_feats = self.fpn(deep_feature, fpn_features)
            hr_feat = ms_feats[-1]
            seg_feats = hr_feat
        return seg_feats, trans_stage_outputs


class TransformerEncoder(nn.Module):

    def __init__(self, num_encoder_layers, nhead, dim_feedforward, d_model, dropout, activation, normalize_before,
                 use_cross_layers, cross_pos, num_classes=1, enc_atpp_branch=1, enc_atpp_res=False, return_intermediate=True):
        super().__init__()
        # ######################################
        # import ipdb; ipdb.set_trace()
        self.cross_pos = cross_pos  # pre, post, cascade
        self.num_layers = num_encoder_layers
        self.num_stages = len(num_encoder_layers)
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        self.return_intermediate = return_intermediate

        for i in range(len(num_encoder_layers)):
            # print('Encoder stage:%d dim_feedforward:%d' % (i, dim_feedforward))
            self.layers.append(nn.ModuleList())
            for j in range(num_encoder_layers[i]):
                _nhead = nhead if i == 0 else 1
                _dim_feedforward = dim_feedforward if i == 0 else d_model
                _dropout = dropout if i == 0 else 0
                _use_layer_norm = (i == 0)
                encoder_layer = TransformerEncoderLayerATPP(d_model, _nhead, _dim_feedforward, _dropout, activation,
                                                            normalize_before, _use_layer_norm, self.num_classes,
                                                            enc_atpp_branch=enc_atpp_branch, enc_atpp_res=enc_atpp_res)
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
        all_intermediates = []
        # import ipdb; ipdb.set_trace()
        for i in range(self.num_stages):
            output = features[i]
            skp_mask = src_key_padding_masks[i]
            pos_embed = pos_embeds[i]
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
                if self.return_intermediate:
                    # import ipdb; ipdb.set_trace()
                    interm = output.clone()
                    s_h, s_w = sizes[i]
                    t_f = interm.shape[0] // (s_h * s_w)
                    interm = interm.permute(1, 2, 0).view(1, 384, t_f, s_h, s_w)  # 1 x C x T x Hx W
                    interm = interm.squeeze(0).permute(1, 0, 2, 3)  # T x C x H x W
                    all_intermediates.append(interm)
            # shape = sizes[i]
            # import ipdb; ipdb.set_trace()
            for j in range(self.num_layers[i]):
                output = self.layers[i][j](output, src_mask=None, src_key_padding_mask=skp_mask, pos=pos_embed,
                                           size=sizes[i])
                if self.return_intermediate:
                    # import ipdb; ipdb.set_trace()
                    interm = output.clone()
                    s_h, s_w = sizes[i]
                    t_f = interm.shape[0] // (s_h * s_w)
                    interm = interm.permute(1, 2, 0).view(1, 384, t_f, s_h, s_w)  # 1 x C x T x Hx W
                    interm = interm.squeeze(0).permute(1, 0, 2, 3)  # T x C x H x W
                    all_intermediates.append(interm)

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
        # Check this later
        if self.norm is not None:
            outputs[0] = self.norm(outputs[0])
            if self.num_stages > 1:
                for i in range(self.num_stages - 1):
                    outputs[i + 1] = self.norms[i](outputs[i + 1])
        return outputs, all_intermediates


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
            memory_size_i = size_list[feat_index]
            output = layer(output, current_dec_feat, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask_i,
                           pos=pos_i, query_pos=query_pos, mem_size=memory_size_i)
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
        # print('Creating cross resolution layer>>> d_model: %3d nhead:%d dim_feedforward:%3d' % (
        #    d_model, nhead, dim_feedforward))
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


class TransformerEncoderLayerATPP(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_layer_norm=True, use_atpp=True, num_classes=1,
                 enc_atpp_branch=1, enc_atpp_res=False):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.nhead = nhead
        self.use_atpp = use_atpp
        self.use_layer_norm = use_layer_norm
        if self.use_atpp:
            self.key_atpp = EncATPP(input_dim=384, model_dim=384, output_dim=384, num_branches=enc_atpp_branch,
                                    groups=4, dropout=0.1,
                                    use_res=enc_atpp_res)
            self.key_norm = nn.LayerNorm(d_model)
            self.region_mem_embed_dim = 256
            self.region_mem_embed = nn.Embedding(self.region_mem_embed_dim, d_model)
            self.key_norm1 = nn.LayerNorm(self.region_mem_embed_dim)
            self.key_norm2 = nn.LayerNorm(d_model)

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

        self.rev_cls_mem_embed = nn.Embedding(num_classes + 2, d_model)
        self.cls_attn = nn.MultiheadAttention(embed_dim=384, num_heads=1, kdim=num_classes + 2, vdim=d_model,
                                              dropout=dropout)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.softmax3 = nn.Softmax(dim=-1)
        self.cls_emb_dropout = nn.Dropout(dropout)
        self.cls_emb_dropout1 = nn.Dropout(dropout)
        self.cls_emb_dropout2 = nn.Dropout(dropout)
        self.cls_emb_norm1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.cls_emb_norm2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.cls_emb_linear1 = nn.Linear(d_model, dim_feedforward)
        self.cls_emb_linear2 = nn.Linear(dim_feedforward, d_model)

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
                     size=None):
        # import ipdb; ipdb.set_trace()
        query = self.with_pos_embed(src, pos)
        value = src
        if self.use_atpp:
            s_h, s_w = size
            t_f = src.shape[0] // (s_h * s_w)
            s_ch = src.shape[-1]
            src_atpp = src.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            src_atpp = src_atpp.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            src_atpp = src_atpp.permute(1, 0, 2, 3).unsqueeze(0)
            src_atpp = self.key_atpp(src_atpp)
            src_atpp = src_atpp.squeeze(0).permute(1, 0, 2, 3)
            src_atpp = src_atpp.reshape(1, t_f, s_ch, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
            src_atpp = src_atpp.flatten(2).permute(2, 0, 1)
            # import ipdb;ipdb.set_trace()
            src_atpp = self.key_norm(src_atpp)
            # #################################################
            src_atpp = torch.matmul(src.permute(1, 0, 2), src_atpp.permute(1, 2, 0)).permute(1, 0, 2)
            # import ipdb;ipdb.set_trace()
            src_atpp = self.softmax1(src_atpp)
            src_atpp = torch.nn.functional.interpolate(src_atpp, size=self.region_mem_embed_dim, mode='linear')
            src_atpp = self.key_norm1(src_atpp)
            src_atpp = torch.matmul(src_atpp, self.region_mem_embed.weight.unsqueeze(0))
            src_atpp = self.softmax2(src_atpp)
            src_atpp = self.key_norm2(src_atpp)
            #################################################
            # import ipdb; ipdb.set_trace()
            # value = key
            key = self.with_pos_embed(src_atpp, pos)
        else:
            key = self.with_pos_embed(src, pos)
            # q = k = self.with_pos_embed(src, pos)
        self_attn_mask = src_mask
        src2 = self.self_attn(query, key, value=value, attn_mask=self_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # import ipdb; ipdb.set_trace()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # ##################################################
        # CLS MEM EMBED ####################
        # import ipdb; ipdb.set_trace()
        # src_cls_emb = torch.matmul(src, self.cls_mem_embed.weight)
        src_cls_emb = torch.matmul(src, self.rev_cls_mem_embed.weight.unsqueeze(0).permute(0, 2, 1))
        src_cls_emb = self.softmax3(src_cls_emb)
        src2, aa = self.cls_attn(query=src, key=src_cls_emb, value=src, attn_mask=self_attn_mask,
                                 key_padding_mask=src_key_padding_mask)
        # import ipdb; ipdb.set_trace()
        src = src + self.cls_emb_dropout(src2)
        src = self.cls_emb_norm1(src)
        src2 = self.cls_emb_linear2(self.cls_emb_dropout1(self.activation(self.cls_emb_linear1(src))))
        src = src + self.cls_emb_dropout2(src2)
        src = self.cls_emb_norm2(src)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, size=None):
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
                size=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, size=size)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, size=size)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_layer_norm=True, use_atpp=True):
        super().__init__()
        self.nhead = nhead
        self.use_atpp = use_atpp
        self.use_layer_norm = use_layer_norm
        if self.use_atpp:
            self.key_atpp = EncATPP(input_dim=384, model_dim=384, output_dim=384, num_branches=2, groups=4, dropout=0.1,
                                    use_res=False)
            self.key_norm = nn.LayerNorm(d_model)
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
                     size=None):
        # import ipdb; ipdb.set_trace()
        query = self.with_pos_embed(src, pos)
        value = src
        if self.use_atpp:
            s_h, s_w = size
            t_f = src.shape[0] // (s_h * s_w)
            s_ch = src.shape[-1]
            key = src.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            key = key.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            key = key.permute(1, 0, 2, 3).unsqueeze(0)
            key = self.key_atpp(key)
            key = key.squeeze(0).permute(1, 0, 2, 3)
            key = key.reshape(1, t_f, s_ch, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
            key = key.flatten(2).permute(2, 0, 1)
            # import ipdb;ipdb.set_trace()
            key = self.key_norm(key)
            key = self.with_pos_embed(key, pos)
        else:
            key = self.with_pos_embed(src, pos)
            # q = k = self.with_pos_embed(src, pos)
        self_attn_mask = src_mask
        src2 = self.self_attn(query, key, value=value, attn_mask=self_attn_mask,
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
                    pos: Optional[Tensor] = None, size=None):
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
                size=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, size=size)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, size=size)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 dec_atpp_branch=0, dec_atpp_res=False):
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
        self.dec_atpp_branch = dec_atpp_branch
        self.dec_atpp_res = dec_atpp_res
        self.use_atpp = self.dec_atpp_branch > 0
        if self.dec_atpp_branch > 0:
            self.key_atpp = EncATPP(input_dim=d_model,
                                    model_dim=d_model,
                                    output_dim=d_model,
                                    num_branches=self.dec_atpp_branch,
                                    groups=4, dropout=0.1,
                                    use_res=self.dec_atpp_res)
            self.key_norm = nn.LayerNorm(d_model)

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
                     mem_size=None):
        # import ipdb;ipdb.set_trace()
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        query_ca = self.with_pos_embed(tgt, query_pos)
        value_ca = memory
        if self.use_atpp:
            s_h, s_w = mem_size
            t_f = memory.shape[0] // (s_h * s_w)
            s_ch = memory.shape[-1]
            src_atpp = memory.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            src_atpp = src_atpp.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            src_atpp = src_atpp.permute(1, 0, 2, 3).unsqueeze(0)
            src_atpp = self.key_atpp(src_atpp)
            src_atpp = src_atpp.squeeze(0).permute(1, 0, 2, 3)
            src_atpp = src_atpp.reshape(1, t_f, s_ch, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
            src_atpp = src_atpp.flatten(2).permute(2, 0, 1)
            src_atpp = self.key_norm(src_atpp)
            key_ca = self.with_pos_embed(src_atpp, pos)
        else:
            key_ca = self.with_pos_embed(memory, pos)
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
                mem_size=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, mem_size=mem_size)


class VOS_SwinMEDVT(nn.Module):
    def __init__(self, args, backbone, backbone_dims, hidden_dim, transformer, num_frames, temporal_strides=[1],
                 n_class=1):
        super().__init__()
        # import ipdb; ipdb.set_trace()
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
            if transformer.decoder_attn_fuse is None or transformer.decoder_attn_fuse == 'cat':
                mask_head_in_channels = hidden_dim + transformer.bbox_nhead  # + 8  # for dec_atpp
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

    def forward(self, samples: NestedTensor, stages=[]):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        if self.training:
            return self._forward_one_samples(samples, stages)
        else:
            return self.forward_inference(samples, stages)

    def forward_inference(self, samples: NestedTensor, stages=[]):
        # import ipdb;ipdb.set_trace()
        samples = self._divide_by_stride(samples)
        all_outs = []
        mean_outs = {}
        with torch.no_grad():
            for stride, samples_ in samples.items():
                all_outs.append(self._forward_one_samples(samples_, stages))
            # import ipdb;ipdb.set_trace()
            mean_preds = torch.stack([a['pred_masks'] for a in all_outs], dim=0).mean(0)
            mean_outs['pred_masks'] = mean_preds
            keys = all_outs[0].keys()
            for key in keys:
                if key != 'pred_masks':
                    mean_outs[key] = all_outs[0][key]
        return mean_outs

    def _forward_one_samples(self, samples: NestedTensor, stages=[]):
        stage_outputs = []
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
            seg_feats, trans_stages = self.transformer(features, pos_list, batch_size)
            stage_outputs = stage_outputs + trans_stages
            mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks = outputs_seg_masks.squeeze(0)

        for i in range(len(stage_outputs)):
            stage_feat = stage_outputs[i]
            stage_outputs[i] = stage_feat.permute(1, 0, 2, 3).flatten(1).permute(1, 0).mean(dim=0, keepdim=True)
        out = {
            "pred_masks": outputs_seg_masks,
            "stage_outputs": stage_outputs
        }
        return out


class VOS_SwinMEDVTLPROP(VOS_SwinMEDVT):
    def __init__(self, args, backbone, backbone_dims, hidden_dim, transformer, num_frames,
                 pretrain_settings={}, lprop_mode=None, temporal_strides=[1], feat_loc=None,
                 stacked=1, n_class=1):
        # import ipdb; ipdb.set_trace()
        super().__init__(args, backbone, backbone_dims, hidden_dim, transformer, num_frames, n_class=n_class)
        logger.debug('Initializing Swin-MEDVT-LPROP')
        # import ipdb;ipdb.set_trace()
        self.stacked = stacked
        self.aux_loss = args.aux_loss > 0

        # from avos.models.atpp import DecATPP
        # self.atpp = DecATPP(input_dim=384, model_dim=128, output_dim=8, num_branches=2)

        if feat_loc == 'early_coarse':
            feat_dim = 384
            hidden_dim = 128
        if feat_loc == 'early_fine':
            feat_dim = 384
            hidden_dim = 128
        elif feat_loc == 'late':
            feat_dim = 384 + transformer.bbox_nhead  # + 8  # ( 8 added for ATPP)
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
        logger.debug('freezing pretrained ...')
        for name, p in self.named_parameters():
            if 'label_propagator' not in name:
                p._requires_grad = False
                p.requires_grad = False

    def _forward_one_samples(self, samples: NestedTensor, stages=[]):
        # import ipdb; ipdb.set_trace()
        # stage_outputs = []
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
            # seg_feats, trans_stages = self.transformer(features, pos_list, batch_size)
            # stage_outputs = stage_outputs + trans_stages
            mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)

        # ##################################################################
        # import ipdb; ipdb.set_trace()
        outputs_seg_masks_b4lprop = self.insmask_head(mask_ins)
        outputs_seg_masks_b4lprop = outputs_seg_masks_b4lprop.squeeze(0)
        outputs_seg_masks_lprop = outputs_seg_masks_b4lprop.permute(1, 0, 2, 3)
        # import ipdb;ipdb.set_trace()
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
        # import ipdb;ipdb.set_trace()
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

        out = {
            "pred_masks": outputs_seg_masks,
            # "stage_outputs": stage_outputs
        }
        if self.aux_loss:
            out['aux_pred_masks'] = outputs_seg_masks_b4lprop
        return out


def build_swin_s_backbone(is_train, _swin_s_pretrained_path):
    logger.debug('creating swin-s-3d backbone>>>')
    swin = SwinTransformer3D(pretrained=None,
                             pretrained2d=True,
                             patch_size=(1, 4, 4),
                             in_chans=3,
                             embed_dim=96,
                             depths=(2, 2, 18, 2),
                             num_heads=(3, 6, 12, 24),
                             window_size=(8, 7, 7),
                             mlp_ratio=4.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_rate=0.,
                             attn_drop_rate=0.,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             patch_norm=True,
                             frozen_stages=-1,
                             use_checkpoint=False)
    if not is_train:  # not need to use backbone initialization during inference
        return swin
    checkpoint = torch.load(_swin_s_pretrained_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'][:, :, 0:1, :, :]
    state_dict['norm_layers.3.weight'] = state_dict['backbone.norm.weight'].clone().detach()
    state_dict['norm_layers.3.bias'] = state_dict['backbone.norm.bias'].clone().detach()
    state_dict['norm_layers.2.weight'] = state_dict['backbone.norm.weight'].clone().detach()
    state_dict['norm_layers.2.bias'] = state_dict['backbone.norm.bias'].clone().detach()
    state_dict['norm_layers.1.weight'] = state_dict['backbone.norm.weight'][:384].clone().detach()
    state_dict['norm_layers.1.bias'] = state_dict['backbone.norm.bias'][:384].clone().detach()
    state_dict['norm_layers.0.weight'] = state_dict['backbone.norm.weight'][:192].clone().detach()
    state_dict['norm_layers.0.bias'] = state_dict['backbone.norm.bias'][:192].clone().detach()
    del state_dict['backbone.norm.weight']
    del state_dict['backbone.norm.bias']
    del state_dict['cls_head.fc_cls.weight']
    del state_dict['cls_head.fc_cls.bias']
    ckpt_keys = [k for k in state_dict.keys()]
    del_keys = []
    for kk in ckpt_keys:
        if 'backbone.' in kk:
            state_dict[kk.replace('backbone.', '')] = state_dict[kk]
            del_keys.append(kk)
    for kk in del_keys:
        del state_dict[kk]
    # logger.debug('len(state_dict): %d' % len(state_dict))
    # logger.debug('len(swin_b.state_dict()): %d' % len(swin.state_dict()))
    swin.load_state_dict(state_dict)
    return swin


def build_swin_b_backbone(is_train, _swin_b_pretrained_path):
    logger.debug('build_swin_b_backbone>>')
    swin = SwinTransformer3D(pretrained=None,
                             pretrained2d=True,
                             patch_size=(1, 4, 4),
                             in_chans=3,
                             embed_dim=128,
                             depths=(2, 2, 18, 2),
                             num_heads=(4, 8, 16, 32),
                             window_size=(8, 7, 7),
                             mlp_ratio=4.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_rate=0.,
                             attn_drop_rate=0.,
                             drop_path_rate=0.1,
                             norm_layer=nn.LayerNorm,
                             patch_norm=True,
                             frozen_stages=-1,
                             use_checkpoint=False)
    if not is_train:  # not need to use backbone initialization during inference
        return swin
    checkpoint = torch.load(_swin_b_pretrained_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'][:, :, 0:1, :, :]
    state_dict['norm_layers.3.weight'] = state_dict['backbone.norm.weight'].clone().detach()
    state_dict['norm_layers.3.bias'] = state_dict['backbone.norm.bias'].clone().detach()
    state_dict['norm_layers.2.weight'] = state_dict['backbone.norm.weight'].clone().detach()
    state_dict['norm_layers.2.bias'] = state_dict['backbone.norm.bias'].clone().detach()
    state_dict['norm_layers.1.weight'] = state_dict['backbone.norm.weight'][:512].clone().detach()
    state_dict['norm_layers.1.bias'] = state_dict['backbone.norm.bias'][:512].clone().detach()
    state_dict['norm_layers.0.weight'] = state_dict['backbone.norm.weight'][:256].clone().detach()
    state_dict['norm_layers.0.bias'] = state_dict['backbone.norm.bias'][:256].clone().detach()
    del state_dict['backbone.norm.weight']
    del state_dict['backbone.norm.bias']
    del state_dict['cls_head.fc_cls.weight']
    del state_dict['cls_head.fc_cls.bias']
    ckpt_keys = [k for k in state_dict.keys()]
    del_keys = []
    for kk in ckpt_keys:
        if 'backbone.' in kk:
            state_dict[kk.replace('backbone.', '')] = state_dict[kk]
            del_keys.append(kk)
    for kk in del_keys:
        del state_dict[kk]
    # logger.debug('len(state_dict): %d' % len(state_dict))
    # logger.debug('len(swin_b.state_dict()): %d' % len(swin.state_dict()))
    # matched_keys = [k for k in state_dict.keys() if k in swin.state_dict().keys()]
    # logger.debug('matched keys:%d' % len(matched_keys))
    swin.load_state_dict(state_dict, strict=False)
    return swin


def build_model_medvt_swinbackbone_without_criterion(args):
    logger.debug('using backbone:%s' % args.backbone)
    # import ipdb;ipdb.set_trace()
    if args.backbone == 'swinS':
        backbone = build_swin_s_backbone(args.is_train, args.swin_s_pretrained_path)
        backbone_dims = (192, 384, 768, 768)
    elif args.backbone == 'swinB':
        backbone = build_swin_b_backbone(args.is_train, args.swin_b_pretrained_path)
        backbone_dims = (256, 512, 1024, 1024)
    else:
        raise ValueError('backbone: %s not implemented!' % args.backbone)
    # logger.debug('args.dim_feedforward:%d' % args.dim_feedforward)
    # import ipdb; ipdb.set_trace()
    if not hasattr(args, 'dec_feat_atpp_branches'):
        args.dec_feat_atpp_branches = 1
    if not hasattr(args, 'dec_feat_atpp_res'):
        args.dec_feat_atpp_res = 1

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
        bbox_nhead=args.bbox_nhead,
        encoder_cross_layer=args.encoder_cross_layer,
        decoder_multiscale=args.dec_multiscale,
        decoder_bbox_head=args.decoder_bbox_head,  # 'tube'/'frame'
        num_classes=args.num_classes,
        enc_atpp_branch=args.enc_atpp_branch,
        enc_atpp_res=args.enc_atpp_res,
        dec_atpp_branch=args.dec_atpp_branch,
        dec_atpp_res=args.dec_atpp_res,
        dec_feat_atpp_branches=args.dec_feat_atpp_branches,
        dec_feat_atpp_res=args.dec_feat_atpp_res,
    )

    if args.is_train:  # weights initialize for training
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
        if args.dec_layers > 6:
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
        # logger.debug('len(shape_mismatch):%d' % len(shape_mismatch))
        for kk in shape_mismatch:
            del checkpoint[kk]
        transformer.load_state_dict(checkpoint, strict=False)
        shape_matched_keys = [k for k in checkpoint.keys() if k in transformer.state_dict().keys()]
        # logger.debug('shape_matched keys:%d' % len(shape_matched_keys))
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

    return model


def build_model_medvt_swinbackbone(args):
    # Model
    model = build_model_medvt_swinbackbone_without_criterion(args)
    # Losses
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    if args.is_train:
        criterion = criterions.SetCriterion(weight_dict=weight_dict, losses=losses, aux_loss=args.aux_loss,
                                            aux_loss_norm=args.aux_loss_norm)
    else:
        criterion = criterions.SetCriterion(weight_dict=weight_dict, losses=losses)
    criterion.to(torch.device(args.device))
    logger.debug('swin model')
    return model, criterion
