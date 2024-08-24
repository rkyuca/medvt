from mmap import mmap
from typing import Optional, List, Any
import copy
import torch
import torchvision.ops
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .misc import NestedTensor
from .utils import expand


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


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
            print('using cross layers...')
            self.cross_res_layers = nn.ModuleList([
                CrossResolutionEncoderLayer(d_model=384, nhead=1, dim_feedforward=384, layer_norm=False,
                                            custom_instance_norm=True, fuse='linear')
            ])

    @staticmethod
    def reshape_up_reshape(src0, src0_shape, src1_shape, batch_size=2, num_frames=5):
        # import ipdb; ipdb.set_trace()
        s_h, s_w = src0_shape
        s_h2, s_w2 = src1_shape
        t_f = src0.shape[0] // (s_h * s_w)
        src0_up = src0.permute(1, 2, 0).view(batch_size, 384, t_f, s_h * s_w)
        src0_up = src0_up.permute(0, 2, 1, 3).reshape(batch_size, t_f, 384, s_h, s_w).flatten(0, 1)
        src0_up = F.interpolate(src0_up, (s_h2, s_w2), mode='bilinear')
        n2, c2, s_h2, s_w2 = src0_up.shape
        src0_up = src0_up.reshape(batch_size, t_f, c2, s_h2, s_w2).permute(0, 2, 1, 3, 4).flatten(-2)
        src0_up = src0_up.flatten(2).permute(2, 0, 1)
        return src0_up

    def forward(self, features, src_key_padding_masks, pos_embeds, sizes, masks=None, pred_mems=None, batch_size=2,
                num_frames=5):
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
                    src0_up = self.reshape_up_reshape(src0, src0_shape=sizes[i - 1], src1_shape=sizes[i],
                                                      batch_size=batch_size, num_frames=num_frames)
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
                output = self.layers[i][j](output, src_mask=None, src_key_padding_mask=skp_mask, pos=pos_embed)
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
                size_list=None,
                ctx_memory: Optional[Tensor] = None,
                ctx_pos: Optional[Tensor] = None,
                ctx_memory_mask: Optional[Tensor] = None,
                ctx_key_padding_mask: Optional[Tensor] = None,
                **kwargs):
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
                           pos=pos_i, query_pos=query_pos,
                           size_list=size_list,
                           ctx_memory=ctx_memory,
                           ctx_pos=ctx_pos,
                           ctx_memory_mask=ctx_memory_mask,
                           ctx_key_padding_mask=ctx_key_padding_mask
                           )
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


class TransformerDecoderMM(nn.Module):

    def __init__(self, num_layers,
                 return_intermediate=False,
                 d_model=384,
                 decoder_nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 mm_context_layers=0):
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayerMM(d_model,
                                                               decoder_nhead,
                                                               dim_feedforward,
                                                               dropout,
                                                               activation,
                                                               normalize_before,
                                                               mm_context= i< mm_context_layers)
                                     for i in range(num_layers)])

    def forward(self, tgt, dec_features,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                size_list=None,
                query_pos_ctx: Optional[Tensor] = None,
                ctx_memory: Optional[Tensor] = None,
                ctx_pos: Optional[Tensor] = None,
                ctx_memory_mask: Optional[Tensor] = None,
                ctx_key_padding_mask: Optional[Tensor] = None,
                **kwargs):
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
                           pos=pos_i, query_pos=query_pos,
                           size_list=size_list,
                           query_pos_ctx=query_pos_ctx,
                           ctx_memory=ctx_memory,
                           ctx_pos=ctx_pos,
                           ctx_memory_mask=ctx_memory_mask,
                           ctx_key_padding_mask=ctx_key_padding_mask
                           )
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
                     pos: Optional[Tensor] = None):
        # import ipdb; ipdb.set_trace()
        q = k = self.with_pos_embed(src, pos)
        self_attn_mask = src_mask
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
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
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
        self.activation = nn.ReLU()
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
        # import ipdb;ipdb.set_trace()
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


class TransformerDecoderLayerMM(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, mm_context=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.use_mm_ctx = mm_context
        if self.use_mm_ctx:
            self.multihead_attn_ctx = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.mm_dropout = nn.Dropout(0.1)
            self.mm_norm = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
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
                     size_list: Optional[Any] = None,
                     query_pos_ctx: Optional[Tensor] = None,
                     ctx_memory: Optional[Tensor] = None,
                     ctx_pos: Optional[Tensor] = None,
                     ctx_memory_mask: Optional[Tensor] = None,
                     ctx_key_padding_mask: Optional[Tensor] = None,
                     ):
        # import ipdb;ipdb.set_trace()
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # #############################
        if self.use_mm_ctx:
            mm_query = self.with_pos_embed(tgt, query_pos_ctx)
            mm_key = self.with_pos_embed(ctx_memory, ctx_pos)
            mm_value = ctx_memory
            tgt2 = self.multihead_attn_ctx(query=mm_query,
                                           key=mm_key,
                                           value=mm_value, attn_mask=ctx_memory_mask,
                                           key_padding_mask=ctx_key_padding_mask)[0]
            tgt = tgt + self.mm_dropout(tgt2)
            tgt = self.mm_norm(tgt)
        # ######################
        query_ca = self.with_pos_embed(tgt, query_pos)
        key_ca = self.with_pos_embed(memory, pos)
        value_ca = memory
        tgt2 = self.multihead_attn(query=query_ca,
                                   key=key_ca,
                                   value=value_ca, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # ##########################
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
                size_list=None,
                query_pos_ctx: Optional[Tensor] = None,
                ctx_memory: Optional[Tensor] = None,
                ctx_pos: Optional[Tensor] = None,
                ctx_memory_mask: Optional[Tensor] = None,
                ctx_key_padding_mask: Optional[Tensor] = None
                ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask,
                                    memory_key_padding_mask,
                                    pos,
                                    query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 size_list=size_list,
                                 query_pos_ctx=query_pos_ctx,
                                 ctx_memory=ctx_memory,
                                 ctx_pos=ctx_pos,
                                 ctx_memory_mask=ctx_memory_mask,
                                 ctx_key_padding_mask=ctx_key_padding_mask
                                 )


class VisFeatPositionEncodings(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=384):
        super(VisFeatPositionEncodings, self).__init__()
        self.d_model = channel
        from .position_encoding import PositionEmbeddingSineFromMask
        self.position_embedding = PositionEmbeddingSineFromMask(self.d_model // 3, num_frames=5, normalize=True)

    def forward(self, visual_features, batch_size, num_frames=5):
        pos_list = []
        for i in range(len(visual_features)):
            x = visual_features[i]
            # import ipdb;ipdb.set_trace()
            _, _, h, w = x.shape  # bt, c , h, w
            mask = torch.zeros((batch_size * num_frames, h, w), dtype=torch.bool, device=x.device)
            pos_list.append(self.position_embedding(mask).to(x.dtype))
        return pos_list


class Context_Encoder(nn.Module):
    def __init__(self, config, args, vis_dim, channel=384):
        super(Context_Encoder, self).__init__()
        # Spatiotemporal Context encoder
        self.vis_dim = vis_dim
        self.d_model = channel
        self.num_encoder_layers = args.vce_enc_layer  # [3, 1]
        self.use_visual_context_encoder = sum(self.num_encoder_layers) > 0
        self.num_encoder_stages = len(self.num_encoder_layers)
        self.encoder_cross_layer = args.vce_use_cross_layer  # False  # hasattr(args, 'encoder_cross_layer') and args.encoder_cross_layer
        if self.num_encoder_stages == 1 and self.encoder_cross_layer:
            self.num_encoder_stages = 2
        if sum(self.num_encoder_layers) > 0:
            self.ce_nhead = args.vce_nhead
            self.dim_feedforward = args.vce_dim_ff
            self.ce_dropout = 0.1
            self.ce_activation = 'relu'
            self.ce_normalize_before = False
            self.num_frames = 5
            self.encoder = TransformerEncoder(self.num_encoder_layers, self.ce_nhead, self.dim_feedforward,
                                              self.d_model, self.ce_dropout,
                                              self.ce_activation,
                                              self.ce_normalize_before, use_cross_layers=self.encoder_cross_layer,
                                              cross_pos='cascade')
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, visual_features, pos_list, batch_size, num_frames=5):
        # BT x C x H x W
        # import ipdb;ipdb.set_trace()
        for i in range(len(visual_features)):
            x = visual_features[i]
            _, _, h, w = x.shape
            mask = torch.zeros((batch_size * num_frames, h, w), dtype=torch.bool, device=x.device)
            x = NestedTensor(x, mask)
            visual_features[i] = x
            # import ipdb;ipdb.set_trace()
        enc_feat_list = []
        enc_mask_list = []
        enc_pos_embd_list = []
        enc_feat_shapes = []
        # import ipdb;ipdb.set_trace()
        for i in range(self.num_encoder_stages):
            feat_index = -1 - i  # taking in reverse order from deeper to shallow
            src_proj, mask = visual_features[feat_index].decompose()
            assert mask is not None
            n, c, s_h, s_w = src_proj.shape
            enc_feat_shapes.append((s_h, s_w))
            src = src_proj.reshape(batch_size, self.num_frames, c, s_h, s_w).permute(0, 2, 1, 3, 4).flatten(-2)
            # bs, c, t, hw = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            enc_feat_list.append(src)
            mask = mask.reshape(batch_size, self.num_frames, s_h * s_w)
            mask = mask.flatten(1)
            enc_mask_list.append(mask)
            pos_embed = pos_list[feat_index].permute(0, 2, 1, 3, 4).flatten(-2)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            enc_pos_embd_list.append(pos_embed)
            # import ipdb;ipdb.set_trace()
            # print('inside enc_features>> i:%d feat_index:%d src.shape:%s mask.shape:%s pos.shape:%s' % (
            # i, feat_index, str(src.shape), str(mask.shape), str(pos_embed.shape)))
        # import ipdb;ipdb.set_trace()
        encoder_features = self.encoder(features=enc_feat_list, src_key_padding_masks=enc_mask_list,
                                        pos_embeds=enc_pos_embd_list, sizes=enc_feat_shapes, batch_size=batch_size,
                                        num_frames=num_frames)
        # import ipdb;ipdb.set_trace()
        for i in range(self.num_encoder_stages):
            memory_i = encoder_features[i]
            h, w = enc_feat_shapes[i]
            memory_i = memory_i.permute(1, 2, 0).view(batch_size, self.d_model, self.num_frames, h * w)
            memory_i = memory_i.permute(0, 2, 1, 3).reshape(batch_size, self.num_frames, self.d_model, h, w)
            memory_i = NestedTensor(memory_i.flatten(0, 1), enc_mask_list[i])
            # import ipdb;ipdb.set_trace()
            visual_features[-1 - i] = memory_i  # bt chw
            # print('enc output>> i:%d  memory_i.shape:%s' % (i, memory_i.shape))
        # import ipdb;ipdb.set_trace()
        # import ipdb;ipdb.set_trace()
        visual_features = [vf.tensors for vf in visual_features]
        return visual_features


class VisualFeatureProj(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, config, args, vis_dim, channel=384):
        super(VisualFeatureProj, self).__init__()
        # Spatiotemporal Context encoder
        self.vis_dim = vis_dim
        self.d_model = channel
        self.input_proj_modules = nn.ModuleList()
        self.input_proj_type = 'conv'  # else
        for backbone_dim in self.vis_dim:
            if self.input_proj_type == 'avs':
                self.input_proj_modules.append(
                    Classifier_Module([3, 6, 12, 18], [3, 6, 12, 18], self.d_model, backbone_dim)
                )
            else:
                self.input_proj_modules.append(
                    nn.Conv2d(backbone_dim, self.d_model, kernel_size=1)
                )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, visual_features, batch_size, num_frames):
        for i in range(len(visual_features)):
            visual_features[i] = self.input_proj_modules[i](visual_features[i])
        return visual_features


class FPN(nn.Module):
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
        # multi_scale_features.append(x)
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


class MMContextEncoderBiDirCrossAttn(nn.Module):
    def __init__(self, config, args, vis_dim=[64, 128, 320, 512], channel=384, context_channel=128):
        super(MMContextEncoderBiDirCrossAttn, self).__init__()
        self.num_stages = len(vis_dim)
        from .bidirectional_cross_attention import BidirectionalCrossAttention
        # self.res_embed = nn.Embedding(1, d_model)
        # res_embed = self.res_embed.weight.unsqueeze(0).repeat(pos_deep.shape[0], 1, 1)
        self.context_pos_embeds = [nn.Embedding(5, channel) for _ in range(self.num_stages)]
        self.context_proj_modules = nn.ModuleList()
        self.attn_modules = nn.ModuleList()
        self.attn_type = 'hw'  # hw vs thw
        for i in range(self.num_stages):
            self.context_proj_modules.append(
                nn.Linear(in_features=context_channel, out_features=channel)
            )
        for i in range(self.num_stages):
            self.attn_modules.append(
                BidirectionalCrossAttention(
                    dim=channel,  # video channels
                    heads=1,
                    context_dim=channel  # audio channels
                )
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature_map_list, audio_features, pos_list, batch_size, num_frames):
        # a_fea_list = [None] * 4
        # import ipdb;ipdb.set_trace()
        for i in range(self.num_stages):
            # import ipdb; ipdb.set_trace()

            joint_cross_attn = self.attn_modules[i]
            video_feat = feature_map_list[i]  # BT x C x H x W
            vbt, vc, vh, vw = video_feat.shape

            audio_feat = audio_features[i]  # .clone().detach()  # BT x C
            audio_feat = self.context_proj_modules[i](audio_feat)
            abt, ac = audio_feat.shape

            if self.attn_type == 'thw':
                video_feat = video_feat.reshape(batch_size, num_frames, vc, vh, vw).permute(0, 2, 1, 3, 4).flatten(
                    2).permute(0, 2, 1)
                video_pos = pos_list[i].clone().detach().permute(0, 2, 1, 3, 4).flatten(
                    2).permute(0, 2, 1).to(video_feat.device)
                video_mask = torch.ones((batch_size, num_frames * vh * vw)).bool().to(video_feat.device)
                # import ipdb;ipdb.set_trace()
                audio_feat = audio_feat.reshape(batch_size, num_frames, ac)
                audio_pos = self.context_pos_embeds[i].weight.unsqueeze(0).repeat(batch_size, 1, 1).to(
                    video_feat.device)
                audio_mask = torch.ones((batch_size, num_frames)).bool().to(video_feat.device)
            else:  # frame-wise
                # import ipdb;ipdb.set_trace()
                video_feat = video_feat.flatten(2).permute(0, 2, 1)
                video_pos = pos_list[i].clone().detach().flatten(0, 1).flatten(2).permute(0, 2, 1).to(video_feat.device)
                video_mask = torch.ones((batch_size * num_frames, vh * vw)).bool().to(video_feat.device)
                # import ipdb;ipdb.set_trace()
                audio_feat = audio_feat.unsqueeze(1)
                audio_pos = self.context_pos_embeds[i].weight.repeat(batch_size, 1).unsqueeze(1).to(video_feat.device)
                audio_mask = torch.ones((batch_size * num_frames, 1)).bool().to(video_feat.device)

            #  import ipdb;ipdb.set_trace()
            # import ipdb;ipdb.set_trace()
            video_out, audio_out = joint_cross_attn(
                video_feat,  # batch, N, c
                audio_feat,  # batch, M, c
                mask=video_mask,
                context_mask=audio_mask,
                pos=video_pos,
                context_pos=audio_pos,
            )
            # import ipdb; ipdb.set_trace()
            if self.attn_type == 'thw':
                video_feat = video_out.permute(0, 2, 1).reshape(batch_size, vc, num_frames, vh, vw).permute(0, 2, 1, 3,
                                                                                                            4).flatten(
                    0, 1)
                audio_feat = audio_out.flatten(0, 1)
                feature_map_list[i] = video_feat
                audio_features[i] = audio_feat
            else:
                video_feat = video_out.permute(0, 2, 1).reshape(batch_size * num_frames, vc, vh, vw)
                audio_feat = audio_out.squeeze(1)
                feature_map_list[i] = video_feat
                audio_features[i] = audio_feat
            # attended output should have the same shape as input
            # assert video_out.shape == video.shape
            # assert audio_out.shape == audio.shape
        return feature_map_list, audio_features


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
        import ipdb;
        ipdb.set_trace()
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        # import ipdb;ipdb.set_trace()
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


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


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class QueryDecoder(nn.Module):
    def __init__(self, num_queries_per_frame=384,
                 num_decoder_layers=12,
                 query_decoder_scales=4,
                 qgen_layers = 3,
                 decoder_head_type='frame',
                 num_frames=5,
                 context_channel=128,
                 decoder_nhead=8,
                 d_model=384,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super().__init__()
        self.num_decoder_layers = num_decoder_layers
        self.query_decoder_scales = query_decoder_scales
        self.num_frames = num_frames
        self.decoder_head_type = decoder_head_type
        self.num_queries_per_frame = num_queries_per_frame
        self.d_model = d_model
        if num_decoder_layers > 0:
            self.query = nn.Embedding(num_queries_per_frame, d_model)
            self.query_pos = nn.Embedding(num_queries_per_frame, d_model)
            self.context_proj = nn.Sequential(
                nn.Linear(in_features=context_channel, out_features=2048),
                nn.ReLU(),
                nn.LayerNorm(2048),
                nn.Linear(in_features=2048, out_features=d_model)
            )
            self.query_initializer = ContextQueryGenerator2(num_layers=qgen_layers, embed_dim=d_model)
            self.decoder = TransformerDecoderMM(num_decoder_layers,
                                                return_intermediate=True,
                                                mm_context_layers= num_decoder_layers // 2)
            self.query_feat_norm = nn.LayerNorm(self.num_queries_per_frame)

    def forward(self, visual_feats, audio_feat, pos_list, batch_size, num_frames):
        # visual_feat ==  high res --- low res
        dec_features = []
        size_list = []
        pos_embed_list = []
        dec_mask_list = []
        # import ipdb;ipdb.set_trace()
        for i in reversed(range(self.query_decoder_scales - 1)):
            fi = visual_feats[i]
            ni, ci, hi, wi = fi.shape
            fi = fi.flatten(2).permute(2, 0, 1)
            dec_mask_i = torch.zeros((batch_size * num_frames, hi * wi)).to(fi.device)
            pe = pos_list[i].clone().detach().to(fi.device).flatten(0, 1).flatten(2).permute(2, 0, 1)
            dec_features.append(fi)
            pos_embed_list.append(pe)
            size_list.append((hi, wi))
            dec_mask_list.append(dec_mask_i)
        # import ipdb;ipdb.set_trace()
        query_pos = self.query_pos.weight.unsqueeze(0).repeat(batch_size * num_frames, 1, 1).permute(1, 0, 2)
        query_pos_ctx = self.query.weight.unsqueeze(0).repeat(batch_size * num_frames, 1, 1).permute(1, 0, 2)

        query = torch.zeros_like(query_pos).to(query_pos.device)*1e-9
        context_feat = self.context_proj(audio_feat.clone().detach().to(query_pos.device)).unsqueeze(
            0)  # .permute(1,0,2)
        query = self.query_initializer(query, context_feat, query_pos=query_pos_ctx, context_pos=None)
        hs = self.decoder(query, dec_features, memory_key_padding_mask=dec_mask_list,
                          pos=pos_embed_list, query_pos=query_pos, size_list=size_list,
                          query_pos_ctx=query_pos_ctx,
                          ctx_memory=context_feat,
                          ctx_pos=None,
                          ctx_memory_mask=None,
                          ctx_key_padding_mask=None,
                          )
        hs = hs.transpose(1, 2)
        hr_feat = visual_feats[0]
        # TODO check
        query_feature = torch.einsum('bqc,bchw->bqhw', hs[-1], hr_feat)
        # import ipdb; ipdb.set_trace()
        b, q, h, w = query_feature.shape
        query_feature = query_feature.flatten(2).permute(0, 2, 1)
        query_feature = self.query_feat_norm(query_feature)
        query_feature = query_feature.permute(0, 2, 1).reshape(b, q, h, w)
        return query_feature


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, audio_feat):
        # import ipdb;ipdb.set_trace()
        out1 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + out1)
        out2 = self.cross_attn(query, audio_feat, audio_feat)[0]
        query = self.norm2(query + out2)
        out3 = self.ffn(query)
        query = self.norm3(query + out3)
        return query


class AttentionQueryGenerator(nn.Module):
    def __init__(self, num_layers, query_num, embed_dim=384, num_heads=8, hidden_dim=1024):
        super().__init__()
        self.num_layers = num_layers
        self.query_num = query_num
        self.embed_dim = embed_dim
        self.query = nn.Embedding(query_num, embed_dim)
        self.layers = nn.ModuleList(
            [AttentionLayer(embed_dim, num_heads, hidden_dim)
             for i in range(num_layers)]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, audio_feat):
        # import ipdb;ipdb.set_trace()
        bs = audio_feat.shape[0]
        query = self.query.weight[None, :, :].repeat(bs, 1, 1)
        for layer in self.layers:
            query = layer(query, audio_feat)
        return query


class ContextQueryGenerator2(nn.Module):
    def __init__(self, num_layers, embed_dim):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_dim)
             for i in range(num_layers)]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, context_feat, query_pos, context_pos):
        for layer in self.layers:
            query = layer(tgt=query, memory=context_feat, query_pos=query_pos, pos=context_pos)
        return query


class Pred_endecoder(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, config, args, channel=384, vis_dim=[64, 128, 320, 512]):
        super(Pred_endecoder, self).__init__()

        self.cfg = config
        self.vis_dim = vis_dim
        self.d_model = channel
        self.visual_backbone = args.visual_backbone

        # import ipdb;ipdb.set_trace()
        if args.visual_backbone == 'swinB':
            from avs.avs_scripts.common.models.swin_transformer_3d import build_swin_b_backbone
            self.encoder_backbone = build_swin_b_backbone()
            self.vis_dim = [256, 512, 1024, 1024]
        else:
            from avs.avs_scripts.common.models.pvt import pvt_v2_b5
            self.encoder_backbone = pvt_v2_b5()

        self.visual_feat_proj = VisualFeatureProj(config, args, self.vis_dim, channel=channel)
        self.visual_feat_poss = VisFeatPositionEncodings(channel=384)
        self.visual_context_encoder = Context_Encoder(config, args, self.vis_dim, channel=channel)
        self.mm_context_encoder = MMContextEncoderBiDirCrossAttn(config, args, self.vis_dim, channel=channel,context_channel=128)
        self.fpn = FPN(channel, channel)

        self.num_decoder_layers = 12
        self.query_decoder_scales = 4
        self.qgen_layers = 3
        task_head_in_channels = self.d_model
        self.decoder_attn_fuse = 'cat'
        if self.num_decoder_layers > 0:
            self.query_decoder = QueryDecoder(
                 num_decoder_layers=self.num_decoder_layers,
                 query_decoder_scales=self.query_decoder_scales,
                 qgen_layers = self.qgen_layers)
            if self.decoder_attn_fuse == 'add':
                task_head_in_channels = self.d_model
                self.query_head_mlp = MLP(self.query_decoder.num_queries_per_frame, 2048, self.d_model, 3)
            elif self.decoder_attn_fuse == 'cat':
                task_head_in_channels = self.query_decoder.num_queries_per_frame + self.d_model  # +
            else:
                task_head_in_channels = self.query_decoder.num_queries_per_frame
        self.output_conv = nn.Sequential(
            #nn.Conv2d(task_head_in_channels, task_head_in_channels, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            nn.Conv2d(task_head_in_channels, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.d_model, 256, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.ReLU(True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=1),
        )
        # if self.training:
        #    self.initialize_pvt_weights()
        if self.training:
            self._reset_parameters()
            if self.visual_backbone == 'swinB':
                self.load_swinb_weights(args.swin_b_pretrained_path)
            else:
                self.initialize_pvt_weights()
            # self.load_context_encoder_weights(args.resnet101_coco_weights_path)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def load_swinb_weights(self, swin_b_pretrained_path):
        ckpt = torch.load(swin_b_pretrained_path, map_location='cpu')
        state_dict = ckpt['state_dict']
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
        self.encoder_backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x, audio_feature=None, batch_size=4, num_frames=5):
        # import ipdb;ipdb.set_trace()
        _, _, i_h, i_w = x.shape
        visual_features = None
        if self.visual_backbone == 'swinB':
            x = x.reshape((batch_size, num_frames, 3, x.shape[2], x.shape[3]))
            x = x.permute(0, 2, 1, 3, 4)
            visual_features = self.encoder_backbone(x)  # B x C x T x H x W
            visual_features = [ff.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous() for ff in visual_features]
            # BT, C, H, W
        elif 'pvt' in self.visual_backbone:  # pvt, pvt-v2
            visual_features = self.encoder_backbone(x)
        else:
            raise ValueError('Backbone not implemented')
        visual_features = self.visual_feat_proj(visual_features, batch_size, num_frames)
        pos_list = self.visual_feat_poss(visual_features, batch_size=batch_size, num_frames=num_frames)
        audio_features = [audio_feature.clone().detach() for _ in range(len(visual_features))]
        visual_features, audio_features = self.mm_context_encoder(visual_features, audio_features, pos_list=pos_list,
                                                                  batch_size=batch_size, num_frames=num_frames)
        visual_features = self.visual_context_encoder(visual_features, pos_list, batch_size)
        visual_features = self.fpn(visual_features[3], list(reversed(visual_features[:3])))[::-1]
        task_head_in = visual_features[0]
        if self.num_decoder_layers > 0:
            obj_attn_masks = self.query_decoder(visual_features, audio_feature, pos_list, batch_size, num_frames)
            if self.decoder_attn_fuse == 'add':
                obj_attn_masks = self.query_head_mlp(obj_attn_masks)
                task_head_in = task_head_in + obj_attn_masks
            elif self.decoder_attn_fuse == 'cat':
                task_head_in = torch.cat([task_head_in, obj_attn_masks], dim=1)
            else:
                task_head_in = obj_attn_masks
        task_head_in = F.interpolate(task_head_in, (i_h // 2, i_w // 2))
        pred = self.output_conv(task_head_in)  # BF x 1 x 224 x 224
        pred = F.interpolate(pred, (i_h, i_w))
        audio_features = [af.reshape(batch_size, num_frames, self.d_model) for af in audio_features]
        return pred, visual_features, audio_features

    def initialize_pvt_weights(self, ):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        # for k, v in pretrained_state_dicts['model'].items():
        #     if k in pvt_model_dict.keys():
        #         print(k, v.requires_grad)
        state_dict = {k: v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
        # pdb.set_trace()



