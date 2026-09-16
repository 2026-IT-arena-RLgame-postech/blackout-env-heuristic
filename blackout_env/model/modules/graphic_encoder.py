import torch
from torch import nn

from blackout_env.model.unit_channels import N_UNIT_CHANNELS, UNIT_GRID_SCALE, UNIT_STEM_CHANNELS

from .bottleneck_block import BottleNeckBlock
from .ffn_block import SwiGLUBlock


class GraphicEncoder(nn.Module):
    def __init__(self, hidden_size : int = 256, in_channels : int = 13 + UNIT_STEM_CHANNELS) -> None:
        super(GraphicEncoder, self).__init__()

        self.hidden_size = hidden_size

        # graphic : [24, 24, 13] = [H, W, C] from MyObsPreprocessor.preprocess_team_graphics
        # (8 base one-hot + 1 battery scalar + 4 item one-hot), concatenated here with 8 channels
        # of unit occupancy. Units used to reach the network only as agent_states coordinate
        # rows, leaving this encoder unable to place them on the map at all.
        #
        # The unit stem takes the 4 occupancy channels MyModel.forward rasterizes at 4x map
        # resolution (96x96, see model/unit_channels.py) and folds each 4x4 block into one map
        # cell with a learned stride-4 conv. Drawing units straight onto 24x24 would round every
        # unit to its tile, and the stack below then pools 24x24 to 6x6 tokens, so sub-tile
        # position would be gone twice over before attention sees anything.
        #
        # Sized down from an earlier 64/128/256-channel version (ImageNet-backbone-scale) after
        # observing grad_norm/graphic_encoder collapse ~7 orders of magnitude within a few
        # thousand steps regardless of two independent fixes (SPR CLS token instead of a
        # mean-pool, and a 100x lower weight_decay on this module alone -- see
        # docs/offline_pretrain_runs.md) -- neither changed the collapse speed at all, pointing
        # to a genuine plateau rather than dilution/erosion. 8 of the 13 input channels (the
        # base terrain/wall one-hot) are constant for an entire episode -- only battery + item
        # one-hots vary tick to tick -- so this input's actual entropy is low relative to what a
        # 256-channel-wide ResNet-style stack is built for; halving channel widths and dropping
        # a block per stage is a bet that a smaller encoder is easier for training to actually
        # rely on (less to get lost in) rather than a capacity increase being what's missing.
        # to be flipped by setting tensor format
        # in -> [B, 24, 24, 13] = [B, H, W, C] + [B, 4, 96, 96] unit occupancy
        self.unit_stem = nn.Sequential(
            nn.Conv2d(N_UNIT_CHANNELS, UNIT_STEM_CHANNELS, kernel_size=UNIT_GRID_SCALE, stride=UNIT_GRID_SCALE),
            nn.SiLU(inplace=True),
        )  # [B, 4, 96, 96] -> [B, 8, 24, 24]

        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2), # -> [B, 32, 24, 24]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> [B, 64, 12, 12]
        )

        self.conv64 = nn.Sequential(  # -> [B, 64, 12, 12]
            BottleNeckBlock(64, 16, 8, 2),
        )
        self.pool = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> [B, 128, 6, 6]

        self.conv128 = nn.Sequential( # -> [B, 128, 6, 6]
            BottleNeckBlock(128, 32, 16, 4),
            BottleNeckBlock(128, 32, 16, 4),
        )
        # -> [B, 128, 36]

        self.tokenize_ffn = SwiGLUBlock(128, 512, self.hidden_size)


    def forward(self, graphic, unit_map):
        """graphic: [B, 13, 24, 24] env channels; unit_map: [B, 4, 96, 96] unit occupancy."""
        x = self.pre_conv(torch.cat((graphic, self.unit_stem(unit_map)), dim=1))
        x = self.conv64(x)
        x = self.pool(x)
        x = self.conv128(x)

        tokens = self.tokenize_ffn(x.flatten(2).transpose(1, 2))
        return tokens