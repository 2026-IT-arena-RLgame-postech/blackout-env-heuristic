import torch
from torch import nn


class BottleNeckBlock(nn.Module):
    """
    Pre-activation residual bottleneck (ResNet-v2 style): x + [GN-SiLU-1x1 down, GN-SiLU-3x3,
    GN-SiLU-1x1 up](x). Shape-preserving, [B, C, H, W] -> [B, C, H, W]. GroupNorm rather than
    BatchNorm so behaviour does not depend on batch size (batch 1 at inference).
    """

    def __init__(
            self,
            input_channels: int,
            bottleneck_channels: int,
            input_norm_groups: int,
            bottleneck_norm_groups: int,
    ) -> None:

        super(BottleNeckBlock, self).__init__()

        self.input_channels = input_channels
        self.bottleneck_channels = bottleneck_channels
        self.input_norm_groups = input_norm_groups
        self.bottleneck_norm_groups = bottleneck_norm_groups

        assert self.bottleneck_channels < self.input_channels, \
            "Bottleneck channels cannot be greater than input channels"
        assert self.input_channels % self.input_norm_groups == 0, \
            "Input channels must be a multiple of input_norm_groups"
        assert self.bottleneck_channels % self.bottleneck_norm_groups == 0, \
            "Bottleneck channels must be a multiple of bottleneck_norm_groups"

        self.block = nn.Sequential (
            # input_channels
            nn.GroupNorm(num_channels = self.input_channels, num_groups = self.input_norm_groups),
            nn.SiLU(inplace=True),
            # input_channels -> bottleneck_channels
            nn.Conv2d(self.input_channels, self.bottleneck_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(num_channels = self.bottleneck_channels, num_groups = self.bottleneck_norm_groups),
            nn.SiLU(inplace=True),
            # bottleneck_channels -> bottleneck_channels
            nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_channels=self.bottleneck_channels, num_groups=self.bottleneck_norm_groups),
            nn.SiLU(inplace=True),
            # bottleneck_channels -> input_channels
            nn.Conv2d(self.bottleneck_channels, self.input_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x) -> torch.Tensor:
        x = x+self.block(x)
        return x
