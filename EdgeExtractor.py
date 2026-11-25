import torch
import torch.nn as nn
import torch.nn.functional as F


def init_fixed_dwconv(conv, kernel_2d):
    """Initialize a fixed 2D kernel for depthwise convolution and freeze its weights."""
    kernel = torch.tensor(kernel_2d, dtype=torch.float32).view(1, 1, 3, 3)
    kernel = kernel.repeat(conv.out_channels, 1, 1, 1)
    with torch.no_grad():
        conv.weight.copy_(kernel)
    conv.weight.requires_grad = False


class SobelEdge(nn.Module):
    """Use Sobel gradients as a structural edge prior."""
    def __init__(self, channels):
        super(SobelEdge, self).__init__()

        self.grad_x = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.grad_y = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )

        sobel_x = [[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]]
        sobel_y = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]

        init_fixed_dwconv(self.grad_x, sobel_x)
        init_fixed_dwconv(self.grad_y, sobel_y)

    def forward(self, x):
        gx = self.grad_x(x)
        gy = self.grad_y(x)
        return torch.abs(gx) + torch.abs(gy)


class LearnableEdge(nn.Module):
    def __init__(self, channels):
        super(LearnableEdge, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class EdgeRefine(nn.Module):
    """Refine edge features with a lightweight residual block."""
    def __init__(self, channels):
        super(EdgeRefine, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.net(x)


class StructureFusion(nn.Module):
    """Use the Sobel prior as the base and inject learned edges through a gate."""
    def __init__(self, channels, hidden_channels=None):
        super(StructureFusion, self).__init__()

        if hidden_channels is None:
            hidden_channels = max(channels // 2, 8)

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.refine = EdgeRefine(channels)

        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, -1.5)

    def forward(self, edge_prior, edge_learned):
        gate = self.gate(torch.cat([edge_prior, edge_learned], dim=1))
        structure = edge_prior + gate * edge_learned
        structure = self.refine(structure)
        return structure, gate


class DenseBlock(nn.Module):
    def __init__(self, channels, negative_slope=0.2):
        super(DenseBlock, self).__init__()
        self.negative_slope = negative_slope

        self.convIO = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2IO = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        out_1 = F.leaky_relu(self.convIO(x), negative_slope=self.negative_slope)
        out_2 = torch.cat((x, out_1), dim=1)

        out_3 = F.leaky_relu(self.conv2IO(out_2), negative_slope=self.negative_slope)
        out_4 = torch.cat((out_2, out_3), dim=1)

        return out_4


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, stride=stride,
            dilation=dilation, groups=groups
        )

    def forward(self, x):
        return self.conv(x)


class StructExtractor(nn.Module):
    def __init__(self, channels):
        super(StructExtractor, self).__init__()

        self.sobel = SobelEdge(channels)
        self.edge = LearnableEdge(channels)
        self.struct_fuse = StructureFusion(channels)

    def forward(self, x):
        edge_prior = self.sobel(x)
        edge_learned = self.edge(x)

        struct_feat, gate = self.struct_fuse(edge_prior, edge_learned)
        return struct_feat


class BFEU(nn.Module):
    """Combine dense texture and Sobel-guided boundary features through a gate."""
    def __init__(self, in_channels, out_channels):
        super(BFEU, self).__init__()

        self.dense = DenseBlock(in_channels)
        self.convdown = PointwiseConv(3 * in_channels, out_channels)

        self.struct_extractor = StructExtractor(in_channels)

        self.struct_up = PointwiseConv(in_channels, out_channels)

        hidden_nc = max(out_channels // 2, 16)

        self.struct_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, hidden_nc, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_nc, out_channels, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

        # Initialize the gate near zero so texture features dominate initially.
        nn.init.zeros_(self.struct_gate[2].weight)
        nn.init.constant_(self.struct_gate[2].bias, -2.0)

    def forward(self, x):
        x_tex = self.dense(x)
        x_tex = self.convdown(x_tex)

        x_struct = self.struct_extractor(x)
        x_struct = self.struct_up(x_struct)

        g_struct = self.struct_gate(torch.cat([x_tex, x_struct], dim=1))

        out = x_tex + g_struct * x_struct
        return F.leaky_relu(out, negative_slope=0.1)