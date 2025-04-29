__all__ = ['PAMNet.py']

# Cell
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from layers.SelfAttention_Family import *
from mamba_ssm import Mamba
from layers.SelfAttention_Family import Attention_Block


class TemporalExternalAttn(nn.Module):
    def __init__(self, d_model, S=256):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries):

        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S

        out = self.mv(attn)  # bs,n,d_model
        return out


class MultiScaleBlock(nn.Module):
    def __init__(self, configs, patch_num, revin = True, affine = True, subtract_last = False):
        super().__init__()
        # self.nvals = configs.enc_in
        # self.pred = configs.pred_len
        self.norm = nn.LayerNorm(configs.d_model)
        # self.norm1 = nn.LayerNorm(configs.h_model)
        # self.stride = configs.stride
        # self.kernel_size = configs.mixer_kernel_size
        # self.h_model = configs.h_model
        # self.dropout = nn.Dropout(configs.dropout)
        # self.lookback = configs.seq_len
        self.d_model = configs.d_model
        # self.patch_num = patch_num
        # self.norm2 = nn.BatchNorm1d(self.d_model)
        self.gelu = nn.GELU()
        # self.conv1d = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        # self.att0 = Attention_Block(configs.d_model, configs.d_ff,
        #                            n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.att1 = TemporalExternalAttn(self.d_model, 512)
        # self.conv2 = PatchMixerLayer(self.d_model, self.d_model, self.kernel_size)
        self.mamba = Mamba(self.d_model, d_state=configs.d_state, d_conv=configs.d_conv)

    def forward(self, x):
        # B, T, N = x.size()
        x_out = self.att1(x)
        # # x_out = self.att0(x)
        x_out = self.norm(x_out)
        x_out = self.gelu(x_out)
        x_out = x_out + x
        x_mamba = self.mamba(x_out)
        return x_mamba


# class Backbone(nn.Module):
#     def __init__(self, configs, revin = True, affine = True, subtract_last = False):
#         super().__init__()
#         self.enc_in = configs.enc_in
#         self.lookback = configs.seq_len
#         self.pred = configs.pred_len
#         self.nvals = configs.enc_in
#         self.patch_size = configs.patch_size
#         self.d_model = configs.d_model
#         self.e_layers = configs.e_layers
#         self.dropout1 = nn.Dropout(0.2)
#         self.h_model = configs.h_model
#         self.stride = configs.stride
#         self.fatten = nn.Flatten(start_dim=-2)
#         self.series_decom = series_decomp(25)
#         self.norm = nn.LayerNorm(configs.d_model)
#         self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
#         self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
#         self.revin = revin
#         if self.revin: self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)
#         self.gelu = nn.GELU()
#         self.mlp1 = nn.Linear(self.patch_size, self.d_model)
#         self.mlp2 = nn.Linear(self.d_model*self.patch_num, int(self.pred * 2))
#         self.mlp3 = nn.Linear(int(self.pred * 2), self.pred)
#         self.block = nn.ModuleList()
#         for _ in range(configs.e_layers):
#             self.block.append(
#                 MultiScaleBlock(configs, self.patch_num, configs.revin, configs.affine, configs.subtract_last)
#             )
#
#
#     def forward(self, x):
#         B, T, N = x.size()
#         if self.revin:
#             x = self.revin_layer(x, 'norm')
#         seasonal_x, trend_x = self.series_decom(x)
#
#         seasonal_x = seasonal_x.permute(0, 2, 1)
#         trend_x = trend_x.permute(0, 2, 1)
#         seasonal_x = self.padding_patch_layer(seasonal_x)
#         seasonal_x = seasonal_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#         seasonal_x = self.mlp1(seasonal_x)
#         seasonal_x = torch.reshape(seasonal_x, (seasonal_x.shape[0] * seasonal_x.shape[1], seasonal_x.shape[2], seasonal_x.shape[3]))
#
#         trend_x = self.padding_patch_layer(trend_x)
#         trend_x = trend_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
#         trend_x = self.mlp1(trend_x)
#         trend_x = torch.reshape(trend_x, (trend_x.shape[0] * trend_x.shape[1], trend_x.shape[2], trend_x.shape[3]))
#         for i in range(self.e_layers):
#             seasonal_x = self.block[i](seasonal_x)
#             trend_x = self.block[i](trend_x)
#         x_out = seasonal_x + trend_x
#         x_out = self.fatten(x_out)
#         x_out = torch.reshape(x_out, (B, N, -1))
#         x_out = self.mlp2(x_out)
#         x_out = self.gelu(x_out)
#         x_out = self.mlp3(x_out)
#         x_out = x_out.permute(0, 2, 1)
#         if self.revin:
#             x_out = self.revin_layer(x_out, 'denorm')
#         return x_out

class Backbone1(nn.Module):
    def __init__(self, configs, revin = True, affine = True, subtract_last = False):
        super().__init__()
        # self.enc_in = configs.enc_in
        self.lookback = configs.seq_len
        self.pred = configs.pred_len
        self.nvals = configs.enc_in
        self.patch_size = configs.patch_size
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        # self.dropout1 = nn.Dropout(0.2)
        # self.h_model = configs.h_model
        self.stride = configs.stride
        self.fatten = nn.Flatten(start_dim=-2)
        # self.series_decom = series_decomp(25)
        self.norm = nn.LayerNorm(configs.d_model)
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)
        self.gelu = nn.GELU()
        self.mlp1 = nn.Linear(self.patch_size, self.d_model)
        self.mlp2 = nn.Linear(self.d_model*self.patch_num, int(self.pred * 2))
        self.mlp3 = nn.Linear(int(self.pred * 2), self.pred)
        self.block = nn.ModuleList()
        for _ in range(configs.e_layers):
            self.block.append(
                MultiScaleBlock(configs, self.patch_num, configs.revin, configs.affine, configs.subtract_last)
            )


    def forward(self, x):
        B, T, N = x.size()
        if self.revin:
            x1 = self.revin_layer(x, 'norm')

        x2 = x1.permute(0, 2, 1)
        x2 = self.padding_patch_layer(x2)
        x2 = x2.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x2 = self.mlp1(x2)
        x2 = torch.reshape(x2, (x2.shape[0] * x2.shape[1], x2.shape[2], x2.shape[3]))
        for i in range(self.e_layers):
            x_encoder = self.block[i](x2)
        x_fatten = self.fatten(x_encoder)
        x_reshape = torch.reshape(x_fatten, (B, N, -1))
        x_out = self.mlp2(x_reshape)
        x_out = self.gelu(x_out)
        x_pred = self.mlp3(x_out)
        x_out = x_pred.permute(0, 2, 1)
        if self.revin:
            x_out = self.revin_layer(x_out, 'denorm')
        return x_out





class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone1(configs)

    def forward(self, x):
        x = self.model(x)
        return x
