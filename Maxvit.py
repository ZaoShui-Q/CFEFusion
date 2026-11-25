from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn

from timm.models._efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.layers import drop_path, trunc_normal_, Mlp, DropPath

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """Accept the arguments passed by timm to activation factories and return GELU."""
    activation = nn.GELU()
    return activation


class MAConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
    ) -> None:
        super(MAConv, self).__init__()
        self.drop_path_rate: float = drop_path
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SimAM(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output



def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    """Split [B, C, H, W] into contiguous windows with channels last. H and W must be divisible by the corresponding window dimensions."""
    B, C, H, W = input.shape
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    """Restore the windowed tensor to [B, C, H, W]."""
    H, W = original_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output

def grid_partition(
        input: torch.Tensor,
        grid_size: Tuple[int, int] = (8,8)
) -> torch.Tensor:
    """Group spatially strided tokens into grids with channels last. H and W must be divisible by the corresponding grid dimensions."""
    B, C, H, W = input.shape
    grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid

def grid_reverse(
        grid: torch.Tensor,
        original_size: Tuple[int, int],
        grid_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    """Restore the grid-partitioned tensor to [B, C, H, W]."""
    (H, W), C = original_size, grid.shape[-1]
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """Build the pairwise relative-position index from the timm Swin V1 implementation."""
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 4,
            grid_window_size: Tuple[int, int] = (8, 8),
            attn_drop: float = 0.,
            drop: float = 0.
    ) -> None:

        super(RelativeSelfAttention, self).__init__()

        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]

        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))

        self.register_buffer("relative_position_index", get_relative_position_index(grid_window_size[0],
                                                                                    grid_window_size[1]))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input: torch.Tensor,
    ) -> torch.Tensor:

        B_, N, C = input.shape
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MaxViTTransformerBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 4,
            grid_window_size: Tuple[int, int] = (8, 8),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super(MaxViTTransformerBlock, self).__init__()
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input.shape
        input_partitioned = self.partition_function(input, self.grid_window_size)
        input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)

        input_norm = self.norm_1(input_partitioned)

        attn_cross = self.attention(input_norm)
        output = input_partitioned + self.drop_path(attn_cross)
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        output = self.reverse_function(output, (H, W), self.grid_window_size)
        return output

class MaxViTBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            num_heads: int = 4,
            grid_window_size: Tuple[int, int] = (8, 8),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        super(MaxViTBlock, self).__init__()

        self.mb1_conv = MAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale=downscale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path
        )
        self.mb2_conv = MAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale=downscale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path
        )

        self.block_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=window_partition,
            reverse_function=window_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )
        self.grid_transformer = MaxViTTransformerBlock(
            in_channels=out_channels,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_ = self.mb1_conv(input)
        x_block_cross = self.block_transformer(x_)
        x_grid_cross = self.grid_transformer(x_block_cross)
        output = x_grid_cross

        return output


class MaxViTStage(nn.Module):
    """Stack MaxViT blocks and downsample in the first block."""

    def __init__(
            self,
            depth: int,
            in_channels: int,
            out_channels: int,
            num_heads: int = 4,
            grid_window_size: Tuple[int, int] = (8, 8),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            norm_layer_transformer: Type[nn.Module] = nn.LayerNorm
    ) -> None:
        super(MaxViTStage, self).__init__()
        self.blocks = nn.Sequential(*[
            MaxViTBlock(
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                downscale=index == 0,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_transformer=norm_layer_transformer
            )
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        output = self.blocks(input)
        return output


class MaxViT(nn.Module):
    """MaxViT classifier based on https://arxiv.org/pdf/2204.01697.pdf."""

    def __init__(
            self,
            in_channels: int = 3,
            depths: Tuple[int, ...] = (2, 2, 5, 2),
            channels: Tuple[int, ...] = (64, 128, 256, 512),
            num_classes: int = 1000,
            embed_dim: int = 64,
            num_heads: int = 4,
            grid_window_size: Tuple[int, int] = (8, 8),
            attn_drop: float = 0.,
            drop=0.,
            drop_path=0.,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_transformer=nn.LayerNorm,
            global_pool: str = "avg"
    ) -> None:
        super(MaxViT, self).__init__()
        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        self.num_classes: int = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            act_layer(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            act_layer(),
        )
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        self.stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            self.stages.append(
                MaxViTStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_transformer=norm_layer_transformer
                )
            )

        self.global_pool: str = global_pool
        self.head = nn.Linear(channels[-1], num_classes)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Exclude relative position bias from weight decay."""
        nwd = set()
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        for stage in self.stages:
            output = stage(output)
        return output

    def forward_head(self, input: torch.Tensor, pre_logits: bool = False):
        """Return pooled features when pre_logits is true."""
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        return input if pre_logits else self.head(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.forward_features(self.stem(input))
        output = self.forward_head(output)
        return output


def max_vit_tiny_224(**kwargs) -> MaxViT:
    return MaxViT(
        depths=(2, 2, 5, 2),
        channels=(64, 128, 256, 512),
        embed_dim=64,
        **kwargs
    )


def max_vit_small_224(**kwargs) -> MaxViT:
    return MaxViT(
        depths=(2, 2, 5, 2),
        channels=(96, 128, 256, 512),
        embed_dim=64,
        **kwargs
    )


def max_vit_base_224(**kwargs) -> MaxViT:
    return MaxViT(
        depths=(2, 6, 14, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        **kwargs
    )


def max_vit_large_224(**kwargs) -> MaxViT:
    return MaxViT(
        depths=(2, 6, 14, 2),
        channels=(128, 256, 512, 1024),
        embed_dim=128,
        **kwargs
    )



