import torch.nn as nn
import math
import torch.nn.functional as F
import torch


############ Utilities
def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=True, relu=True, padding_mode='zeros'):
    layers = []
    assert padding_mode == 'zeros' or padding_mode == 'replicate'

    if padding_mode == 'replicate' and padding > 0:
        assert isinstance(padding, int)
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ########## Label Propagator:
class GraphAttention(nn.Module):
    def __init__(self, feature_dim=512, key_dim=128, value_dim=None, tau=1 / 30, topk=False, no_learning=False):
        super(GraphAttention, self).__init__()

        self.no_learning = no_learning
        if value_dim is None:
            value_dim = feature_dim
        if not no_learning:
            self.WK = nn.Linear(feature_dim, key_dim)
            # self.WQ = nn.Linear(feature_dim, key_dim)
            self.WV = nn.Linear(value_dim, value_dim)

            # Init weights
            for m in self.WK.modules():
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
            """
            for m in self.WQ.modules():
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
            """

            for m in self.WV.modules():
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()

        self.tau = tau
        self.topk = topk

    def forward(self, query=None, key=None, value=None):
        # import ipdb;ipdb.set_trace()
        if not self.no_learning:
            w_k = self.WK(key)
            # w_q = self.WQ(query)
            w_q = self.WK(query)
        else:
            w_k = key
            w_q = query

        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0)  # Batch, Dim, Len_1

        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2)  # Batch, Len_2, Dim

        w_v = self.WV(value)
        # w_v = value
        w_v = w_v.permute(1, 0, 2)  # Batch, Len_1, Dim

        dot_prod = torch.bmm(w_q, w_k)  # Batch, Len_2, Len_1
        if self.topk:
            affinity = softmax_topk(dot_prod)
        else:
            affinity = F.softmax(dot_prod / self.tau, dim=-1)

        output = torch.bmm(affinity, w_v)  # Batch, Len_2, Dim
        output = output.permute(1, 0, 2)  # Len_2, Batch, Dim

        return output


class MaskedGraphAttention(GraphAttention):
    def __init__(self, feature_dim=512, key_dim=128, tau=1 / 30, topk=False, no_learning=False):
        super(MaskedGraphAttention, self).__init__(feature_dim=feature_dim, key_dim=key_dim,
                                                   value_dim=feature_dim,
                                                   tau=tau, topk=topk, no_learning=no_learning)

    def forward(self, query=None, key=None, value=None, mask=None):
        assert mask is not None, "Cant use masked attention without mask"
        if not self.no_learning:
            w_k = self.WK(key)
            # w_q = self.WQ(query)
            w_q = self.WK(query)
        else:
            w_k = key
            w_q = query

        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0)  # Batch, Dim, Len_1

        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2)  # Batch, Len_2, Dim

        # w_v = self.WV(value)
        w_v = value
        w_v = w_v.permute(1, 0, 2)  # Batch, Len_1, Dim

        dot_prod = torch.bmm(w_q, w_k)  # Batch, Len_2, Len_1
        dot_prod = dot_prod + mask.unsqueeze(0)
        if self.topk:
            affinity = softmax_topk(dot_prod)
        else:
            affinity = F.softmax(dot_prod / self.tau, dim=-1)

        output = torch.bmm(affinity, w_v)  # Batch, Len_2, Dim
        output = output.permute(1, 0, 2)  # Len_2, Batch, Dim
        return output


class LPropDecoderLayer(nn.Module):
    def __init__(self, feature_dim, key_dim):
        super().__init__()
        self.self_attn = GraphAttention(feature_dim=feature_dim, key_dim=key_dim, value_dim=feature_dim)
        self.cross_attn = GraphAttention(feature_dim=feature_dim, key_dim=key_dim, value_dim=16, tau=1 / 30,
                                         topk=False)

        self.norm = nn.InstanceNorm2d(feature_dim)

    def instance_norm(self, src, input_shape):
        num_frames, num_sequences, c, h, w = input_shape
        # Normlization
        src = src.reshape(num_frames, h, w, num_sequences, c).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, c, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, num_sequences, c)
        return src

    def forward(self, tgt, memory, pos_enc=None):
        """
        tgt:    1, num_sequences, c, h, w
        memory: num_frames, num_sequences, c, h, w
        pos_enc:num_frames, num_sequences, ce, h, w
        """
        # import ipdb;ipdb.set_trace()
        tgt_shape = tgt.shape
        mem_shape = memory.shape
        num_frames, num_sequences, c, h, w = mem_shape

        tgt = tgt.reshape(1, num_sequences, c, -1).permute(0, 3, 1, 2)
        tgt = tgt.reshape(-1, num_sequences, c)

        memory = memory.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        memory = memory.reshape(-1, num_sequences, c)

        if pos_enc is not None:
            pos_enc = pos_enc.reshape(num_frames, num_sequences, -1, h * w).permute(0, 3, 1, 2)
            pos_enc = pos_enc.reshape(num_frames * h * w, num_sequences, -1)

        # self-attention
        tgt_attn = self.self_attn(query=tgt, key=tgt, value=tgt)
        tgt = tgt + tgt_attn

        tgt = self.instance_norm(tgt, tgt_shape)

        # ## Mask Encoding transform
        enc = self.cross_attn(query=tgt, key=memory, value=pos_enc)
        out = enc.reshape(1, h, w, num_sequences, -1).permute(0, 3, 4, 1, 2)
        return out  # .sigmoid()


class MaskedLPropDecoderLayer(nn.Module):
    def __init__(self, feature_dim, key_dim, lprop_mode):
        super().__init__()
        self.cross_attn = MaskedGraphAttention(feature_dim=feature_dim, key_dim=key_dim, tau=1 / 30,
                                               topk=False, no_learning=(lprop_mode == 3))

        if lprop_mode != 3:
            self.self_attn = MaskedGraphAttention(feature_dim=feature_dim, key_dim=key_dim, tau=1 / 30,
                                                  topk=False, no_learning=(lprop_mode == 3))
            self.norm = nn.InstanceNorm2d(feature_dim)

        self.lprop_mode = lprop_mode

    def instance_norm(self, src, input_shape):
        num_frames, num_sequences, c, h, w = input_shape
        # Normlization
        src = src.reshape(num_frames, h, w, num_sequences, c).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, c, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, num_sequences, c)
        return src

    def forward(self, tgt, memory, pos_enc=None):
        """
        tgt:    num_frames, num_sequences, c, h, w
        memory: num_frames, num_sequences, c, h, w
        pos_enc:num_frames, num_sequences, ce, h, w
        """
        # import ipdb;ipdb.set_trace()
        tgt_shape = tgt.shape
        mem_shape = memory.shape
        pos_enc_shape = pos_enc.shape
        num_frames, num_sequences, c, h, w = mem_shape

        mask = torch.eye(num_frames).to(tgt.device)  # torch.zeros((num_frames, h, w, num_frames, h, w)).to(tgt.device)
        mask[mask == 1] = -1 * float("Inf")
        mask = mask.view(num_frames, 1, 1, num_frames, 1, 1).repeat(1, h, w, 1, h, w)
        # mask2 = pos_enc.squeeze(1).squeeze(1) > 0.5
        # mask = mask + mask2.view(*mask2.shape, 1, 1, 1).repeat(1, 1, 1, *mask2.shape[:3])
        mask = mask.view(num_frames * h * w, num_frames * h * w)

        tgt = tgt.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        tgt = tgt.reshape(-1, num_sequences, c)

        memory = memory.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        memory = memory.reshape(-1, num_sequences, c)

        if pos_enc is not None:
            pos_enc = pos_enc.reshape(num_frames, num_sequences, -1, h * w).permute(0, 3, 1, 2)
            pos_enc = pos_enc.reshape(num_frames * h * w, num_sequences, -1)

        if self.lprop_mode != 3:
            # self-attention
            tgt_attn = self.self_attn(query=tgt, key=tgt, value=tgt, mask=mask)
            tgt = tgt + tgt_attn
            tgt = self.instance_norm(tgt, tgt_shape)

        # ## Mask Encoding transform
        enc = self.cross_attn(query=tgt, key=memory, value=pos_enc, mask=mask)
        # import ipdb; ipdb.set_trace()
        enc = self.instance_norm(enc, pos_enc_shape)
        out = enc.reshape(num_frames, h, w, num_sequences, -1).permute(0, 3, 4, 1, 2)
        # if self.lprop_mode != 3:
        #    out = out.sigmoid()
        return out


class MaskedLPropDecoderLayerV1(nn.Module):
    def __init__(self, feature_dim, key_dim, lprop_mode):
        super().__init__()
        self.cross_attn = MaskedGraphAttention(feature_dim=feature_dim, key_dim=key_dim, tau=1 / 30,
                                               topk=False, no_learning=(lprop_mode == 3))

        if lprop_mode != 3:
            self.self_attn = GraphAttention(feature_dim=feature_dim, key_dim=key_dim)
            self.norm = nn.InstanceNorm2d(feature_dim)

        self.lprop_mode = lprop_mode

    def instance_norm(self, src, input_shape):
        num_frames, num_sequences, c, h, w = input_shape
        # Normlization
        src = src.reshape(num_frames, h, w, num_sequences, c).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, c, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, num_sequences, c)
        return src

    def forward(self, tgt, memory, pos_enc=None):
        """
        tgt:    num_frames, num_sequences, c, h, w
        memory: num_frames, num_sequences, c, h, w
        pos_enc:num_frames, num_sequences, ce, h, w
        """
        # import ipdb;ipdb.set_trace()
        tgt_shape = tgt.shape
        mem_shape = memory.shape
        pos_enc_shape = pos_enc.shape
        num_frames, num_sequences, c, h, w = mem_shape

        mask = torch.eye(num_frames).to(tgt.device)  # torch.zeros((num_frames, h, w, num_frames, h, w)).to(tgt.device)
        mask[mask == 1] = -1 * float("Inf")
        mask = mask.view(num_frames, 1, 1, num_frames, 1, 1).repeat(1, h, w, 1, h, w)
        # mask2 = pos_enc.squeeze(1).squeeze(1) > 0.5
        # mask = mask + mask2.view(*mask2.shape, 1, 1, 1).repeat(1, 1, 1, *mask2.shape[:3])
        mask = mask.view(num_frames * h * w, num_frames * h * w)

        tgt = tgt.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        tgt = tgt.reshape(-1, num_sequences, c)

        memory = memory.reshape(num_frames, num_sequences, c, -1).permute(0, 3, 1, 2)
        memory = memory.reshape(-1, num_sequences, c)

        if pos_enc is not None:
            pos_enc = pos_enc.reshape(num_frames, num_sequences, -1, h * w).permute(0, 3, 1, 2)
            pos_enc = pos_enc.reshape(num_frames * h * w, num_sequences, -1)

        if self.lprop_mode != 3:
            # self-attention
            tgt_attn = self.self_attn(query=tgt, key=tgt, value=tgt)
            tgt = tgt + tgt_attn
            tgt = self.instance_norm(tgt, tgt_shape)

        # ## Mask Encoding transform
        enc = self.cross_attn(query=tgt, key=memory, value=pos_enc, mask=mask)
        # import ipdb; ipdb.set_trace()
        enc = self.instance_norm(enc, pos_enc_shape)
        out = enc.reshape(num_frames, h, w, num_sequences, -1).permute(0, 3, 4, 1, 2)
        # if self.lprop_mode != 3:
        #    out = out.sigmoid()

        return out


class LabelPropagator(nn.Module):
    def __init__(self, lprop_mode, feat_dim=392, hidden_dim=128, label_scale=16, n_class=80):
        super().__init__()
        self.lprop_mode = lprop_mode
        self.lprop_scale = label_scale
        lbl_enc_dims = [16, 32, 64, 16]
        if lprop_mode == 1:
            self.prop_layer = LPropDecoderLayer(feat_dim, hidden_dim)
        else:
            self.prop_layer = MaskedLPropDecoderLayer(feat_dim, hidden_dim, lprop_mode=lprop_mode)

        self.label_encoder = LabelEncoder(lbl_enc_dims)
        self.label_decoder = nn.Sequential(
            nn.Conv3d(lbl_enc_dims[-1], lbl_enc_dims[-1], (3, 3, 3), padding='same'),
            nn.GroupNorm(4, lbl_enc_dims[-1]),
            nn.ReLU(),
            nn.Conv3d(lbl_enc_dims[-1], n_class, (1, 1, 1), padding='same')
        )

    def forward(self, masks, feats):
        # original_shape = masks.shape[-2:]
        masks = masks.permute(1, 0, 2, 3).unsqueeze(0)
        original_shape = masks.shape
        # import ipdb; ipdb.set_trace()
        # TODO: Use the learned mask encoding weights to reweight the loss
        up_shape = [ int(s * self.lprop_scale) if idx > 2 else s for idx, s in enumerate(original_shape)]
        up_shape = up_shape[-3:]

        masks = F.interpolate(masks, up_shape)
        masks_enc, _ = self.label_encoder(masks)
        feats = F.interpolate(feats, masks_enc.shape[-2:])
        feats = feats.unsqueeze(0)

        if self.lprop_mode == 1:
            # Unidirectional Label Propagation
            prev_feats = feats[:, :-1].permute(1, 0, 2, 3, 4)
            curr_feats = feats[:, -1:].permute(1, 0, 2, 3, 4)
            prev_masks_enc = masks_enc[:, :-1].permute(1, 0, 2, 3, 4)
            curr_masks_enc = self.prop_layer(curr_feats, prev_feats, prev_masks_enc)
            curr_avg_masks_enc = torch.mean(torch.cat([masks_enc[:, -1:], curr_masks_enc], dim=1), dim=1)
            new_masks_enc = torch.cat([masks_enc[:, :-1], curr_avg_masks_enc.unsqueeze(1)], dim=1)
            new_masks_enc = new_masks_enc.permute(0, 2, 1, 3, 4)
        elif self.lprop_mode == 2:
            # Bidirectional Label Propagation
            new_masks_enc = self.prop_layer(feats.permute(1, 0, 2, 3, 4),
                                            feats.permute(1, 0, 2, 3, 4),
                                            masks_enc.permute(1, 0, 2, 3, 4))
            new_masks_enc = new_masks_enc.permute(1, 2, 0, 3, 4)

        # masks = self.label_decoder(new_masks_enc).sigmoid()
        masks = self.label_decoder(new_masks_enc)
        masks = masks.squeeze(0)
        return masks


# ########## Label Encoder used from: https://github.com/maoyunyao/JOINT/blob/main/ltr/models/joint/label_encoder.py

class LabelEncoder(nn.Module):
    """ Outputs the few-shot learner label and spatial importance weights given the segmentation mask """

    def __init__(self, layer_dims, use_bn=True, n_cls=80):
        super().__init__()
        # self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1,
        #                              batch_norm=False)
        self.conv_block = conv_block(80, layer_dims[0], kernel_size=3, stride=2, padding=1,
                                     batch_norm=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1_tm = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1_tm = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1_tm,
                                  use_bn=False)
        ds2_tm = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2_tm = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2_tm,
                                  use_bn=use_bn)

        self.label_pred_tm = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                        relu=True, batch_norm=False)
        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask):
        # label_mask: frames, seq, h, w
        # assert label_mask.dim() == 4

        # label_mask: [batch (1), num_class, num_frames, h, w]

        label_shape = label_mask.shape
        # label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])
        label_mask = label_mask.squeeze().permute(1, 0, 2, 3)  # [num_frames, num_class, h, w]

        out = self.pool(self.conv_block(label_mask))
        out_tm = self.res2_tm(self.res1_tm(out))

        label_enc_tm = self.label_pred_tm(out_tm)
        sample_w_tm = self.samp_w_pred(out_tm)

        label_enc_tm = label_enc_tm.view(label_shape[0], label_shape[2], *label_enc_tm.shape[-3:])
        sample_w_tm = sample_w_tm.view(label_shape[0], label_shape[2], *sample_w_tm.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc_tm, sample_w_tm
