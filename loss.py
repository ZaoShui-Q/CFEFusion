#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim

ms_ssim = pytorch_msssim.msssim


class Fusionloss(nn.Module):
    """Combine structural consistency, foreground detail, and background suppression.

    The auxiliary outputs follow the training interface. loss_in is the detail loss,
    ssim_value is the structural loss, and loss_grad is the background loss.
    """

    def __init__(
        self,
        lambda_ssim=1.0,
        lambda_fg=10.0,
        lambda_bg=0.3,
        ssim_weight_vis=0.6,
        win_size=5,
        clip_for_ssim=True,
        eps=1e-6,
        charbonnier_eps=1e-3,
    ):
        super(Fusionloss, self).__init__()

        self.lambda_ssim = lambda_ssim
        self.lambda_fg = lambda_fg
        self.lambda_bg = lambda_bg

        self.ssim_weight_vis = ssim_weight_vis 
        self.ssim_weight_ir = 1 - ssim_weight_vis 

        self.win_size = win_size
        self.clip_for_ssim = clip_for_ssim

        self.eps = eps
        self.charbonnier_eps = charbonnier_eps

        self.sobel = Sobelxy()

    def forward(self, image_vis, image_ir, fused):

        fused_ssim = torch.clamp(fused, 0.0, 1.0) if self.clip_for_ssim else fused
        loss_ssim_1 = 1.0 - ms_ssim(fused_ssim, image_vis, normalize=True)
        loss_ssim_2 = 1.0 - ms_ssim(fused_ssim, image_ir, normalize=True)
        loss_ssim = (
            self.ssim_weight_vis * loss_ssim_1
            + self.ssim_weight_ir * loss_ssim_2
        )

        weight = self.make_weight(image_vis, image_ir)

        grad1 = self.compress(self.sobel(image_vis))
        grad2 = self.compress(self.sobel(image_ir))
        grad_fused = self.compress(self.sobel(fused))

        std1 = self.compress(self.local_std(image_vis))
        std2 = self.compress(self.local_std(image_ir))
        std_fused = self.compress(self.local_std(fused))

        grad_target = self.mix_target(grad1, grad2)
        std_target = self.mix_target(std1, std2)

        loss_grad_detail = self.masked_charbonnier(grad_fused, grad_target, weight)
        loss_std_detail = self.masked_charbonnier(std_fused, std_target, weight)
        loss_detail = 0.5 * (loss_grad_detail + loss_std_detail)

        bg_weight = 1.0 - weight
        bg_target = torch.minimum(image_vis, image_ir)
        loss_bg = self.masked_charbonnier(fused, bg_target, bg_weight)

        loss_total = (
            self.lambda_ssim * loss_ssim
            + self.lambda_fg * loss_detail
            + self.lambda_bg * loss_bg
        )

        loss_in = loss_detail
        ssim_value = loss_ssim
        loss_grad = loss_bg

        return loss_total, loss_in, ssim_value, loss_grad


    def blur(self, x):
        if self.win_size <= 1:
            return x
        pad = self.win_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(x, kernel_size=self.win_size, stride=1, padding=0)

    def local_std(self, x):
        mean = self.blur(x)
        mean_sq = self.blur(x * x)
        var = torch.clamp(mean_sq - mean * mean, min=0.0)
        return torch.sqrt(var + self.eps)

    def compress(self, x):
        """Compress large responses logarithmically to reduce their influence."""
        return torch.log1p(torch.clamp(x, min=0.0))

    def soft_map(self, x):
        """Map per-image standardized responses to [0, 1]."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        return torch.sigmoid((x - mean) / (std + self.eps))

    def masked_charbonnier(self, pred, target, weight):
        diff = pred - target
        loss_map = torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps)
        num = torch.sum(weight * loss_map)
        den = torch.sum(weight) + self.eps
        return num / den

    def mix_target(self, x1, x2):
        x_max = torch.maximum(x1, x2)
        x_mean = 0.5 * (x1 + x2)
        return 0.5 * x_max + 0.5 * x_mean

    def make_weight(self, image_vis, image_ir):
        """Build foreground weights from equally averaged intensity and local contrast responses."""
        inten = 0.5 * (
            self.soft_map(torch.clamp(image_vis, min=0.0)) +
            self.soft_map(torch.clamp(image_ir, min=0.0))
        )

        detail = 0.5 * (
            self.soft_map(self.local_std(image_vis)) +
            self.soft_map(self.local_std(image_ir))
        )

        weight = 0.5 * inten + 0.5 * detail
        weight = self.blur(weight)

        return torch.clamp(weight, 0.0, 1.0)


class Sobelxy(nn.Module):
    """Compute Sobel gradients with reflection padding."""
    def __init__(self):
        super(Sobelxy, self).__init__()

        kernel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        kernel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("weight_x", kernel_x)
        self.register_buffer("weight_y", kernel_y)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(x, self.weight_x, padding=0)
        grad_y = F.conv2d(x, self.weight_y, padding=0)
        return torch.abs(grad_x) + torch.abs(grad_y)