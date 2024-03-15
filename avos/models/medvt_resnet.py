"""
MED-VT model.
Some parts of the code is taken from VisTR (https://github.com/Epiphqny/VisTR)
which was again Modified from DETR (https://github.com/facebookresearch/detr)
And modified as needed.
"""
import torchvision
import torchvision.ops
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import logging
from avos.utils.misc import is_main_process
from avos.utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from avos.models import criterions
from avos.models.label_propagation import LabelPropagator
from avos.models.position_encoding import build_position_encoding
from avos.models.utils import get_clones, get_activation_fn, expand
from avos.models.mh_attention_map import MHAttentionMap

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # print('dilation:{}'.format(dilation))
        if not type(dilation) is list:
            dilation = [False, False, dilation]
        # print('dilation:{}'.format(dilation))
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=dilation,
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.is_train
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
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
        # self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.conv_offset = torch.nn.Conv2d(inter_dims[3], 18, 1)  # , bias=False)
        self.dcn = torchvision.ops.DeformConv2d(inter_dims[3], inter_dims[4], 3, padding=1, bias=False)
        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

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

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        multi_scale_features.append(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # dcn for the last layer
        offset = self.conv_offset(x)
        x = self.dcn(x, offset)
        # x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)
        return x, multi_scale_features


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, num_encoder_stages=2, num_decoder_stages=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layers = []
        for i in range(num_encoder_stages):
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward // (2 ** i),
                                                    dropout, activation, normalize_before)
            encoder_layers.append(encoder_layer)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.mask_head = MaskHeadSmallConv(384, [384, 512, 256], 384)

        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, s_h, s_w, s_h2, s_w2, t_f, src, mask, query_embed, pos_embed, src2, mask2, pos2, features,
                pos_list):

        bs, c, t, hw = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        bs2, c2, t2, hw2 = src2.shape
        src2 = src2.flatten(2).permute(2, 0, 1)
        pos2 = pos2.flatten(2).permute(2, 0, 1)
        mask2 = mask2.flatten(1)
        for i in range(3):
            _, c_f, h, w = features[i].tensors.shape
            features[i].tensors = features[i].tensors.reshape(bs, t, c_f, h, w)
        memory, memory2 = self.encoder(s_h, s_w, s_h2, s_w2, t_f, src, src_key_padding_mask=mask, pos=pos_embed,
                                       src2=src2, mask2=mask2, pos2=pos2)
        memory = memory.permute(1, 2, 0).view(bs, c, t, hw)
        memory = memory.permute(0, 2, 1, 3).reshape(bs, t, c, s_h, s_w).flatten(0, 1)

        memory2 = memory2.permute(1, 2, 0).view(bs2, c2, t2, hw2)
        memory2 = memory2.permute(0, 2, 1, 3).reshape(bs2, t2, c2, s_h2, s_w2).flatten(0, 1)
        # import ipdb;ipdb.set_trace()

        hr_feat, ms_feats = self.mask_head(memory,
                                           [memory2, features[1].tensors.flatten(0, 1),
                                            features[0].tensors.flatten(0, 1)])
        dec_features = []
        pos_embed_list = []
        size_list = []
        dec_mask_list = []
        for i in range(3):
            fi = ms_feats[i]
            ni, ci, hi, wi = fi.shape
            fi = fi.reshape(bs, t, ci, hi, wi).permute(0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)
            dec_mask_i = features[-1 - i].mask.reshape(bs, t, hi * wi).flatten(1)
            pe = pos_list[-1 - i].permute(0, 2, 1, 3, 4).flatten(-2).flatten(2).permute(2, 0, 1)
            dec_features.append(fi)
            pos_embed_list.append(pe)
            size_list.append((hi, wi))
            dec_mask_list.append(dec_mask_i)
        query_embed = query_embed.unsqueeze(1)
        tq, bq, cq = query_embed.shape
        query_embed = query_embed.repeat(t_f // tq, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, dec_features, memory_key_padding_mask=dec_mask_list,
                          pos=pos_embed_list, query_pos=query_embed, size_list=size_list)
        return hs.transpose(1, 2), hr_feat


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder : TransformerEncoderWorking Original
    """

    def __init__(self, encoder_layers, num_layers, norm=None, cross_pos=0):
        super().__init__()
        self.num_layers = num_layers
        self.num_stages = len(encoder_layers)
        layer_list = [copy.deepcopy(encoder_layers[0]) for _ in range(6)]
        layer_list = layer_list + [copy.deepcopy(encoder_layers[1]) for _ in range(1)]
        self.layers = nn.ModuleList(layer_list)
        self.norm = norm
        self.cross_pos = cross_pos
        self.norm2 = None if norm is None else copy.deepcopy(norm)
        self.cross_res_layers = nn.ModuleList([CrossResolutionEncoderLayer(384, 8, 1024)])

    def forward(self, s_h, s_w, s_h2, s_w2, t_f, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src2=None, mask2=None, pos2=None):
        shape_src = (t_f, s_h, s_w)
        shape_src2 = (t_f, s_h2, s_w2)

        output = src
        output2 = src2
        if self.cross_pos == -1 or self.cross_pos == 2:
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        for i in range(6):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos,
                                    shapes=shape_src)
        if self.cross_pos == 0 or self.cross_pos == 2:
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        for i in range(6, 7):
            output2 = self.layers[i](output2, src_mask=None, src_key_padding_mask=mask2, pos=pos2, shapes=shape_src2)
        if self.cross_pos == 1 or self.cross_pos == 2:
            outputx = output.permute(1, 2, 0).view(1, 384, t_f, s_h * s_w)
            outputx = outputx.permute(0, 2, 1, 3).reshape(1, t_f, 384, s_h, s_w).flatten(0, 1)
            outputx = F.interpolate(outputx, (s_h2, s_w2), mode='bilinear')
            n2, c2, s_h2, s_w2 = outputx.shape
            outputx = outputx.reshape(1, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
            outputx = outputx.flatten(2).permute(2, 0, 1)
            output2 = self.cross_res_layers[0](outputx, output2, src_key_padding_mask, mask2, pos, pos2)
        if self.norm is not None:
            output = self.norm(output)
            output2 = self.norm2(output2)
        return output, output2


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
                size_list=None):
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

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu", custom_norm=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.custom_norm = custom_norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, src1, src2, mask1, mask2, pos1, pos2):
        kk = src1
        if self.custom_norm:
            ku = src1.mean(dim=0)
            ks = src1.std(dim=0)
            qu = src2.mean(dim=0)
            qs = src2.std(dim=0)
            kk = (kk - ku)
            if ks.min() > 1e-10:
                kk = kk / ks
            kk = (kk * qs) + qu
        kp = kk + pos2
        qp = src2 + pos2
        attn_mask = torch.mm(mask2.transpose(1, 0).double(), mask2.double()).bool()
        out = self.self_attn(qp, kp, value=kk, attn_mask=attn_mask, key_padding_mask=mask2)[0]
        out = src2 + self.dropout1(out)
        out = self.norm1(out)
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout2(out2)
        out = self.norm2(out)
        return out


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_atpp_keys=False):
        super().__init__()
        self.use_atpp_keys = False  # use_atpp_keys  # default False
        if self.use_atpp_keys:
            # from models.atpp import ATPP
            # self.atpp = ATPP(input_dim=384, model_dim=384, output_dim=384, num_branches=2, groups=4, dropout=0.1)
            self.atpp = nn.Sequential(
                nn.Conv3d(384, 384, (3, 3, 3), padding='same', dilation=(2, 1, 1)),
                # nn.GroupNorm(4, 384),
                # nn.ReLU()
            )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
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
                     shapes=None):
        # import ipdb;ipdb.set_trace()
        if self.use_atpp_keys:
            thw, bs, c = src.shape
            t, s_h, s_w = shapes
            hw = s_h * s_w
            kk = src.permute(1, 2, 0).view(bs, c, t, hw)
            kk = kk.permute(0, 2, 1, 3).reshape(bs, t, c, s_h, s_w).flatten(0, 1)
            kk = self.atpp(kk.unsqueeze(0).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).squeeze(0)
            n, c, s_h, s_w = kk.shape
            kk = kk.reshape(bs, t, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
            bs, c, t, hw = kk.shape
            kk = kk.flatten(2).permute(2, 0, 1)
            k = self.with_pos_embed(kk, pos)
        else:
            k = self.with_pos_embed(src, pos)
        q = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
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
                shapes=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, shapes=shapes)


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


class VisTR(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = transformer.d_model
        self.num_frames = num_frames
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.input_proj_2 = nn.Conv2d(backbone.num_channels // 2, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, samples: NestedTensor):
        pass


class MEDVT(nn.Module):
    def __init__(self, vistr, freeze_vistr=False, temporal_strides=[1], n_class=1):
        super().__init__()
        # import ipdb; ipdb.set_trace()
        self.backbone_name = 'resnet101'
        self.num_frames = vistr.num_frames
        self.vistr = vistr
        if freeze_vistr:
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim, nheads = vistr.transformer.d_model, vistr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, num_heads=nheads, dropout=0.0)
        in_channel = nheads + hidden_dim
        self.insmask_head = nn.Sequential(
            nn.Conv3d(in_channel, 384, (1, 3, 3), padding='same', dilation=1),
            nn.GroupNorm(4, 384),
            nn.ReLU(),
            nn.Conv3d(384, 256, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.Conv3d(256, 128, 3, padding='same', dilation=2),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv3d(128, n_class, 1))

        self.temporal_strides = temporal_strides

    def _divide_by_stride(self, samples: NestedTensor):
        samples_dict = {}
        for it, stride in enumerate(self.temporal_strides):
            start = it * self.num_frames
            end = (it + 1) * self.num_frames
            samples_dict[stride] = NestedTensor(samples.tensors[start:end], samples.mask[start:end])
        return samples_dict

    def forward(self, samples: NestedTensor, clip_tag=None, **kwargs):
        if self.training:
            return self._forward_one_samples(samples, clip_tag)
        else:
            return self.forward_inference(samples, clip_tag)

    def forward_inference(self, samples: NestedTensor, clip_tag=None):
        samples = self._divide_by_stride(samples)
        all_outs = []
        with torch.no_grad():
            for stride, samples_ in samples.items():
                all_outs.append(self._forward_one_samples(samples_, clip_tag=clip_tag))
            all_outs = torch.stack([a['pred_masks'] for a in all_outs], dim=0)
        return {'pred_masks': all_outs.mean(0)}

    def _forward_one_samples(self, samples: NestedTensor, clip_tag=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos_list = self.vistr.backbone(samples)  # ## check warnings on floor_divide
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        ####################################
        src_proj = self.vistr.input_proj(src)
        n, c, s_h, s_w = src_proj.shape
        bs_f = bs // self.vistr.num_frames
        src_proj = src_proj.reshape(bs_f, self.vistr.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
        mask = mask.reshape(bs_f, self.vistr.num_frames, s_h * s_w)
        pos = pos_list[-1].permute(0, 2, 1, 3, 4).flatten(-2)
        src_2, mask_2 = features[-2].decompose()
        assert mask_2 is not None
        src_proj_2 = self.vistr.input_proj_2(src_2)
        n2, c2, s_h2, s_w2 = src_proj_2.shape
        src_proj_2 = src_proj_2.reshape(bs_f, self.vistr.num_frames, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
        mask_2 = mask_2.reshape(bs_f, self.vistr.num_frames, s_h2 * s_w2)
        pos_2 = pos_list[-2].permute(0, 2, 1, 3, 4).flatten(-2)
        hs, hr_feat = self.vistr.transformer(s_h, s_w, s_h2, s_w2, self.vistr.num_frames, src_proj, mask,
                                             self.vistr.query_embed.weight, pos, src_proj_2, mask_2, pos_2, features,
                                             pos_list)
        n_f = 1 if self.vistr.num_queries == 1 else self.vistr.num_queries // self.vistr.num_frames
        obj_attn_masks = []
        for i in range(self.vistr.num_frames):
            t2, c2, h2, w2 = hr_feat.shape
            hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
            memory_f = hr_feat[i, :, :, :].reshape(bs_f, c2, h2, w2)
            mask_f = features[0].mask[i, :, :].reshape(bs_f, h2, w2)
            obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
            obj_attn_masks.append(obj_attn_mask_f)
        obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
        seg_feats = torch.cat([hr_feat, obj_attn_masks], dim=1)
        mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)

        outputs_seg_masks = outputs_seg_masks.squeeze(0)
        out = {"pred_masks": outputs_seg_masks}
        return out


class MEDVT_LPROP(MEDVT):
    def __init__(self, args, vistr, freeze_vistr=False,
                 pretrain_settings={}, lprop_mode=None, temporal_strides=[1], feat_loc=None, stacked=1, n_class=1):
        super().__init__(vistr, freeze_vistr, n_class=n_class)
        # import ipdb; ipdb.set_trace()
        logger.debug('Building model -> MEDVT_LPROP')
        self.stacked = stacked
        self.num_frames = args.num_frames
        if feat_loc == 'early_coarse':
            feat_dim = 384
            hidden_dim = 128
        if feat_loc == 'early_fine':
            feat_dim = 384
            hidden_dim = 128
        elif feat_loc == 'late':
            feat_dim = 384 + 8
            hidden_dim = 128
        elif feat_loc == 'attmaps_only':
            feat_dim = 8
            hidden_dim = 16

        self.lprop_mode = lprop_mode
        self.label_propagator = LabelPropagator(lprop_mode, feat_dim=feat_dim, hidden_dim=hidden_dim,
                                                label_scale=args.lprop_scale, n_class=n_class)
        self.feat_loc = feat_loc
        self.temporal_strides = temporal_strides

        # import ipdb; ipdb.set_trace()
        # Pretraining Settings
        if pretrain_settings is None:
            pretrain_settings = {}

        if 'freeze_pretrained' not in pretrain_settings:
            pretrain_settings['freeze_pretrained'] = False
            # pretrain_settings['pretrained_model_path'] = ''

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
        """
        pretrained_modules = [self.vistr.backbone, self.vistr.transformer]
        for mod in pretrained_modules:
            for p in mod.parameters():
                p._requires_grad = False
                p.requires_grad = False
        """
        for name, p in self.named_parameters():
            if 'label_propagator' not in name:
                p._requires_grad = False
                p.requires_grad = False

    def _forward_one_samples(self, samples: NestedTensor, clip_tag=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos_list = self.vistr.backbone(samples)  # ## check warnings on floor_divide
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.vistr.input_proj(src)
        n, c, s_h, s_w = src_proj.shape
        bs_f = bs // self.vistr.num_frames
        src_proj = src_proj.reshape(bs_f, self.vistr.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
        mask = mask.reshape(bs_f, self.vistr.num_frames, s_h * s_w)
        pos = pos_list[-1].permute(0, 2, 1, 3, 4).flatten(-2)
        src_2, mask_2 = features[-2].decompose()
        assert mask_2 is not None
        src_proj_2 = self.vistr.input_proj_2(src_2)
        n2, c2, s_h2, s_w2 = src_proj_2.shape
        src_proj_2 = src_proj_2.reshape(bs_f, self.vistr.num_frames, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
        mask_2 = mask_2.reshape(bs_f, self.vistr.num_frames, s_h2 * s_w2)
        pos_2 = pos_list[-2].permute(0, 2, 1, 3, 4).flatten(-2)
        hs, hr_feat = self.vistr.transformer(s_h, s_w, s_h2, s_w2, self.vistr.num_frames, src_proj, mask,
                                             self.vistr.query_embed.weight, pos, src_proj_2, mask_2, pos_2, features,
                                             pos_list)
        n_f = 1 if self.vistr.num_queries == 1 else self.vistr.num_queries // self.vistr.num_frames
        obj_attn_masks = []
        for i in range(self.vistr.num_frames):
            t2, c2, h2, w2 = hr_feat.shape
            hs_f = hs[-1][:, i * n_f:(i + 1) * n_f, :]
            memory_f = hr_feat[i, :, :, :].reshape(bs_f, c2, h2, w2)
            mask_f = features[0].mask[i, :, :].reshape(bs_f, h2, w2)
            obj_attn_mask_f = self.bbox_attention(hs_f, memory_f, mask=mask_f).flatten(0, 1)
            obj_attn_masks.append(obj_attn_mask_f)
        obj_attn_masks = torch.cat(obj_attn_masks, dim=0)
        seg_feats = torch.cat([hr_feat, obj_attn_masks], dim=1)
        mask_ins = seg_feats.unsqueeze(0).permute(0, 2, 1, 3, 4)
        outputs_seg_masks = self.insmask_head(mask_ins)
        outputs_seg_masks_medvt = outputs_seg_masks.squeeze(0)
        outputs_seg_masks_lprop = outputs_seg_masks_medvt.permute(1, 0, 2, 3)

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

        if self.lprop_mode in [1, 2]:
            if sum(outputs_seg_masks_lprop.shape[-2:]) < sum(outputs_seg_masks_medvt.shape[-2:]):
                outputs_seg_masks_lprop = F.interpolate(outputs_seg_masks_lprop, outputs_seg_masks_medvt.shape[-2:])
            elif sum(outputs_seg_masks_lprop.shape[-2:]) > sum(outputs_seg_masks_medvt.shape[-2:]):
                outputs_seg_masks_medvt = F.interpolate(outputs_seg_masks_medvt, outputs_seg_masks_lprop.shape[-2:])
            outputs_seg_masks = torch.stack([outputs_seg_masks_medvt, outputs_seg_masks_lprop], dim=0).mean(0)
        elif self.lprop_mode == 3:
            outputs_seg_masks = outputs_seg_masks_lprop
        out = {
            "pred_masks": outputs_seg_masks,
            'aux_pred_masks': outputs_seg_masks_medvt
        }
        return out


def initi_pretrained_weights(args, model):
    checkpoint = torch.load(args.resnet101_coco_weights_path, map_location='cpu')['model']
    model_keys = model.state_dict().keys()
    # import ipdb;
    # ipdb.set_trace()
    if args.dec_layers > 6 and not args.finetune:
        cks = [k for k in checkpoint.keys() if 'vistr.transformer.decoder.layers.5.' in k]
        for i in range(6, args.dec_layers):
            for ck in cks:
                mk = ck.replace('5', str(i))
                checkpoint[mk] = checkpoint[ck].clone().detach()
    # import ipdb;
    # ipdb.set_trace()
    # if 'vistr.transformer.encoder.cross_res_layers.0.':
    cks = [k for k in checkpoint.keys() if 'vistr.transformer.encoder.layers.0.' in k]
    for ck in cks:
        # if 'norm' in ck:
        #    continue
        mk = ck.replace('layers.0.', 'cross_res_layers.0.')
        checkpoint[mk] = checkpoint[ck].clone().detach()
        # mk = ck.replace('layers.0.', 'cross_res_layers.1.')
        # checkpoint[mk] = checkpoint[ck].clone().detach()
    if int(args.enc_layers) > 6:
        additional_lyrs = int(args.enc_layers) - 6
        for i in range(additional_lyrs):
            cks = [k for k in checkpoint.keys() if 'vistr.transformer.encoder.layers.%d.' % i in k]
            for ck in cks:
                # mk = ck.replace('%d' % (5-i), '%d' % (5+additional_lyrs-i))
                mk = ck.replace('%d' % i, '%d' % (i + 6))
                checkpoint[mk] = checkpoint[ck].clone().detach()
    # import ipdb;ipdb.set_trace()
    #####################
    del_keys = []
    for k in checkpoint.keys():
        if k not in model_keys:
            del_keys.append(k)
            continue
        if checkpoint[k].shape != model.state_dict()[k].shape:
            del_keys.append(k)
            continue
    for k in del_keys:
        del checkpoint[k]
    model.load_state_dict(checkpoint, strict=False)


def build_model_without_criterion(args):
    if type(args.enc_layers) in [list, tuple]:
        args.enc_layers = sum(args.enc_layers)

    backbone = build_backbone(args)
    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_encoder_stages=2,
        num_decoder_stages=3,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
    # taken from VisTR and used this structure for the convenience of using the same coco pretrained model
    vistr = VisTR(
        backbone,
        transformer,
        num_classes=1,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
    )
    if args.lprop_mode > 0:
        temporal_strides = [1] if not hasattr(args, 'temporal_strides') else args.temporal_strides
        model = MEDVT_LPROP(args,
                            vistr,
                            pretrain_settings=args.pretrain_settings,
                            lprop_mode=args.lprop_mode,
                            temporal_strides=temporal_strides,
                            feat_loc=args.feat_loc,
                            stacked=args.stacked_lprop,
                            n_class=args.num_classes)
    else:
        model = MEDVT(vistr, n_class=args.num_classes)
    ####################################################################################
    return model

def build_model(args):
    model = build_model_without_criterion(args)
    if args.is_train and hasattr(args, 'resnet101_coco_weights_path') and args.resnet101_coco_weights_path is not None:
        initi_pretrained_weights(args, model)
    ###########################################################################################
    weight_dict = {"loss_mask": args.mask_loss_coef, "loss_dice": args.dice_loss_coef}
    losses = ["masks"]
    if args.is_train:
        criterion = criterions.SetCriterion(weight_dict=weight_dict, losses=losses, aux_loss=args.aux_loss,
                                            aux_loss_norm=args.aux_loss_norm)
    else:
        criterion = criterions.SetCriterion(weight_dict=weight_dict, losses=losses)
    criterion.to(torch.device(args.device))
    logger.debug('built resnet model')
    return model, criterion
