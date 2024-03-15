import torch
import torch.nn as nn


class ATPP(nn.Module):
    def __init__(self, input_dim=384, model_dim=384, output_dim=384, num_branches=1, groups=4, dropout=0.1):
        super(ATPP, self).__init__()

        self.branches = nn.ModuleList()
        for ii in range(1, num_branches + 1):
            branch = nn.Sequential(
                nn.Conv3d(input_dim, model_dim//num_branches, (3, 3, 3), padding='same', dilation=(ii, 1, 1)),
                nn.GroupNorm(groups, model_dim//num_branches),
                nn.ReLU())
            self.branches.add_module('branch_%d' % ii, branch)

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels=input_dim, out_channels= output_dim, kernel_size=(1, 1, 1), padding='same', dilation=(1, 1, 1)),
            nn.GroupNorm(groups, model_dim),
            nn.ReLU())

        self.out_proj = nn.Sequential(
            nn.Conv3d(model_dim, output_dim, (1, 1, 1), padding='same', dilation=(1, 1, 1)),
            nn.GroupNorm(groups, output_dim),
            nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        # self._init_weight()

    def forward(self, x):
        import ipdb;ipdb.set_trace()
        branch_outs = []
        for branch in self.branches:
            bo = branch(x)
            branch_outs.append(bo)
        bo = torch.cat(branch_outs, dim=1)
        short = self.shortcut(x)
        out = self.out_proj(self.dropout(bo))
        out = out+short
        return out


class EncATPP(nn.Module):
    def __init__(self, input_dim=384, model_dim=384, output_dim=384, num_branches=1, groups=4, dropout=0.1, use_res=True):
        super(EncATPP, self).__init__()
        self.use_res = use_res
        self.branches = nn.ModuleList()
        # import ipdb; ipdb.set_trace()
        for ii in range(1, num_branches + 1):
            branch = nn.Sequential(
                nn.Conv3d(in_channels= input_dim, out_channels=model_dim, kernel_size=(3, 3, 3), padding='same', dilation=(ii+1, 1, 1)),
                nn.GroupNorm(groups, model_dim),
                nn.ReLU())
            self.branches.add_module('branch_%d' % ii, branch)
        if self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels= input_dim, out_channels= output_dim, kernel_size=(1, 1, 1), padding='same', dilation=(1, 1, 1)),
                nn.GroupNorm(groups, output_dim),
                nn.ReLU())

        self.out_proj = nn.Sequential(
            nn.Conv3d(in_channels= model_dim*num_branches, out_channels=output_dim, kernel_size=(1, 1, 1), padding='same', dilation=(1, 1, 1)),
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
    def __init__(self, input_dim=384, model_dim=384, output_dim=384, num_branches=1, groups=4, dropout=0.1):
        super(DecATPP, self).__init__()

        self.branches = nn.ModuleList()
        for ii in range(1, num_branches + 1):
            branch = nn.Sequential(
                nn.Conv3d(in_channels= input_dim, out_channels=model_dim, kernel_size=(3, 3, 3), padding='same', dilation=(ii, 1, 1)),
                nn.GroupNorm(groups, model_dim),
                nn.ReLU())
            self.branches.add_module('branch_%d' % ii, branch)

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels= input_dim, out_channels=output_dim, kernel_size= (1, 1, 1), padding='same', dilation=(1, 1, 1)),
            nn.GroupNorm(groups, output_dim),
            nn.ReLU())

        self.out_proj = nn.Sequential(
            nn.Conv3d(in_channels= model_dim*num_branches,out_channels= output_dim, kernel_size=(1, 1, 1), padding='same', dilation=(1, 1, 1)),
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
        short = self.shortcut(x)
        out = self.out_proj(self.dropout(bo))
        out = out + short
        return out
