# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
Reference: https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class encoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=None, naf_dim=32, naf_depth=6):
        super(encoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dim = naf_dim
        self.depth = naf_depth

        self.down0 = nn.Sequential(
                    nn.Conv2d(self.input_channels, self.dim, kernel_size=3, stride=1, padding=1, groups=1, bias=True),
                    NAFBlock(self.dim)
                )
        self.down1 = nn.Sequential(
                    nn.Conv2d(self.dim, 2*self.dim, 2, 2),
                    NAFBlock(2*self.dim)
                )
        self.down2 = nn.Sequential(
                    nn.Conv2d(2*self.dim, 4*self.dim, 2, 2),
                    NAFBlock(4*self.dim)
                )
        self.down3 = [nn.Conv2d(4*self.dim, 8*self.dim, 2, 2)]
        for _ in range(self.depth):
            self.down3.append(NAFBlock(8*self.dim))
        self.down3 = nn.Sequential(*self.down3)
        
        if self.output_channels is not None:
            self.output_proj = nn.Conv2d(8*self.dim, self.output_channels, 1)

    def forward(self, x):
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        if self.output_channels is not None:
            x = self.output_proj(x)
        return x

class decoder(nn.Module):
    def __init__(self, input_channels=None, output_channels=1, naf_dim=32, naf_depth=6):
        super(decoder, self).__init__()
        self.output_channels = output_channels
        self.dim = naf_dim
        self.depth = naf_depth
        self.input_channels = input_channels

        if self.input_channels is not None:
            self.input_proj = nn.Conv2d(self.input_channels, 8*self.dim, 1)

        self.up32 = []
        for _ in range(self.depth):
            self.up32.append(NAFBlock(8*self.dim))
        self.up32.append(nn.Conv2d(8*self.dim, 16*self.dim, 1))
        self.up32.append(nn.PixelShuffle(2))
        self.up32 = nn.Sequential(*self.up32)

        self.up64 =  nn.Sequential(
                    NAFBlock(4*self.dim),
                    nn.Conv2d(4*self.dim, 8*self.dim, 1),
                    nn.PixelShuffle(2)
                )

        self.up128 =  nn.Sequential(
                    NAFBlock(2*self.dim),
                    nn.Conv2d(2*self.dim, 4*self.dim, 1),
                    nn.PixelShuffle(2)
                )
  
        self.up256 = nn.Sequential(
                    NAFBlock(self.dim),
                    nn.Conv2d(self.dim, self.output_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
                )

    def forward(self, x):
        if self.input_channels is not None:
            x = self.input_proj(x)

        x = self.up32(x)
        x = self.up64(x)
        x = self.up128(x)
        x = self.up256(x)

        return x # when ddp training, tanh() may lead to NAN with larger learning rate. remove it. However, we need more training iterations.

class concat_conv(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        super(concat_conv, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim_out = dim_out

        self.conv = nn.Conv2d(self.dim1+self.dim2, self.dim_out, 1)
    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))

class decoder_2(nn.Module):
    def __init__(self, code_dim, output_channels=1, naf_dim=32, naf_depth=6):
        super(decoder_2, self).__init__()
        self.output_channels = output_channels
        self.dim = naf_dim
        self.code_dim = code_dim
        self.depth = naf_depth

        self.proj_fusing = concat_conv(self.code_dim, 8*self.dim, 16*self.dim)

        self.up32 = []
        for _ in range(self.depth):
            self.up32.append(NAFBlock(16*self.dim))
        self.up32.append(nn.Conv2d(16*self.dim, 16*self.dim, 1))
        self.up32.append(nn.PixelShuffle(2))
        self.up32 = nn.Sequential(*self.up32)

        self.up64 =  nn.Sequential(
                    NAFBlock(4*self.dim),
                    nn.Conv2d(4*self.dim, 8*self.dim, 1),
                    nn.PixelShuffle(2)
                )

        self.up128 =  nn.Sequential(
                    NAFBlock(2*self.dim),
                    nn.Conv2d(2*self.dim, 4*self.dim, 1),
                    nn.PixelShuffle(2)
                )

        self.up256 = nn.Sequential(
                    NAFBlock(self.dim),
                    nn.Conv2d(self.dim, self.output_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
                )

    def forward(self, x1, x2):
        x = self.proj_fusing(x1, x2)
        x = self.up32(x)
        x = self.up64(x)
        x = self.up128(x)
        x = self.up256(x)
        return x # when ddp training, tanh() may lead to NAN with larger learning rate. remove it. However, we need more training iterations.