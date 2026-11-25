import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from CrossMaxvit import MaxViTBlock as crossmax
from Maxvit import MaxViTBlock as Max
from EdgeExtractor import BFEU


# Intensity-preserving stem
class IPS(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(IPS, self).__init__()

        self.base = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=bias
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1,
                groups=out_channels, bias=bias
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=1, stride=1, padding=0, bias=bias
            )
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Sigmoid()
        )

        nn.init.zeros_(self.refine[2].weight)
        if self.refine[2].bias is not None:
            nn.init.zeros_(self.refine[2].bias)

    def forward(self, x):
        x_main = self.base(x)
        x_detail = self.refine(x_main)
        g = self.gate(x_main)
        out = x_main + g * x_detail
        return out

class LSAM(nn.Module):
    def __init__(self, nf=48, window_size=7, use_residual=True, eps=1e-5):
        super(LSAM, self).__init__()

        assert window_size % 2 == 1, "window_size must be odd (for example, 5, 7, or 9)"

        self.window_size = window_size
        self.use_residual = use_residual
        self.eps = eps

        self.conv_shared = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
            nn.GELU()
        )
        self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

    def local_stats(self, x):
        k = self.window_size

        mean = F.avg_pool2d(
            x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False
        )
        mean_sq = F.avg_pool2d(
            x * x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False
        )
        var = torch.clamp(mean_sq - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)

        return mean, std

    def forward(self, x1, x2):
        x1_mean, x1_std = self.local_stats(x1)
        x2_mean, x2_std = self.local_stats(x2)

        x2_normed = (x2 - x2_mean) / (x2_std + self.eps)

        style = torch.cat([x1, x2], dim=1)
        style = self.conv_shared(style)

        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)

        if self.use_residual:
            gamma = gamma + x1_std
            beta = beta + x1_mean

        out = x2_normed * gamma + beta
        return out


class Select(nn.Module):
    def __init__(self, in_nc=48, hidden_nc=None):
        super(Select, self).__init__()

        if hidden_nc is None:
            hidden_nc = max(in_nc // 2, 16)

        self.mask_net = nn.Sequential(
            nn.Conv2d(in_nc * 2, hidden_nc, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_nc, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, complement):
        mask = self.mask_net(torch.cat([x, complement], dim=1))
        out = mask * complement
        return out


class LSAMIM(nn.Module):
    """Local-statistics-aligned cross-modal interaction module."""
    def __init__(self, in_dim, window_size=7):
        super(LSAMIM, self).__init__()

        self.lsam = LSAM(nf=in_dim, window_size=window_size)
        self.crossmax = crossmax(in_dim, in_dim)
        self.select = Select(in_nc=in_dim)

    def forward(self, x1, x2):
        x2_aligned = self.lsam(x1, x2)
        complement = self.crossmax(x1, x2_aligned)
        complement = self.select(x1, complement)
        x_out = x1 + complement
        return x_out

class DecodeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(DecodeBlock, self).__init__()
        pad = int(np.floor(kernel_size / 2))
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = F.leaky_relu(out, negative_slope=0.1)
        return out


class Fusion(nn.Module):
    def __init__(self, output=1):
        super(Fusion, self).__init__()

        func_ch = [16, 32, 48, 64]
        stru_ch = [16, 32, 48, 64]

        self.func_ips = IPS(1, func_ch[0])
        self.func_bfeu1 = BFEU(func_ch[0], func_ch[1])
        self.func_bfeu2 = BFEU(func_ch[1], func_ch[2])
        self.func_bfeu3 = BFEU(func_ch[2], func_ch[3])
        self.func_self = Max(func_ch[3], func_ch[3])
        self.func_lsamim = LSAMIM(func_ch[3])

        self.stru_ips = IPS(1, stru_ch[0])
        self.stru_bfeu1 = BFEU(stru_ch[0], stru_ch[1])
        self.stru_bfeu2 = BFEU(stru_ch[1], stru_ch[2])
        self.stru_bfeu3 = BFEU(stru_ch[2], stru_ch[3])
        self.stru_self = Max(stru_ch[3], stru_ch[3])
        self.stru_lsamim = LSAMIM(stru_ch[3])

        self.decoder = nn.Sequential(
            DecodeBlock(2 * func_ch[3], func_ch[3], 1, 1),
            DecodeBlock(func_ch[3], 32, 3, 1),
            DecodeBlock(32, output, 3, 1)
        )

    def forward(self, image_func, image_stru):
        func_in = image_func[:, :1]
        stru_in = image_stru

        func_feat0 = self.func_ips(func_in)
        func_feat1 = self.func_bfeu1(func_feat0)
        func_feat2 = self.func_bfeu2(func_feat1)
        func_feat3 = self.func_bfeu3(func_feat2)
        func_feat4 = self.func_self(func_feat3)

        stru_feat0 = self.stru_ips(stru_in)
        stru_feat1 = self.stru_bfeu1(stru_feat0)
        stru_feat2 = self.stru_bfeu2(stru_feat1)
        stru_feat3 = self.stru_bfeu3(stru_feat2)
        stru_feat4 = self.stru_self(stru_feat3)

        func_fused = self.func_lsamim(func_feat4, stru_feat4)
        stru_fused = self.stru_lsamim(stru_feat4, func_feat4)

        fused_feat = torch.cat((func_fused, stru_fused), dim=1)
        out = self.decoder(fused_feat)

        return out
