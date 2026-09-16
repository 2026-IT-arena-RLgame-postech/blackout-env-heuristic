from torch import nn

from blackout_env.model.derived_obs import N_DERIVED_MAP_CHANNELS

from .bottleneck_block import BottleNeckBlock
from .ffn_block import SwiGLUBlock


class GraphicEncoder(nn.Module):
    def __init__(self, hidden_size : int = 256, in_channels : int = 13 + N_DERIVED_MAP_CHANNELS) -> None:
        super(GraphicEncoder, self).__init__()

        self.hidden_size = hidden_size

        # graphic : [24, 24, 19] = [H, W, C] -- 13 channels from
        # MyObsPreprocessor.preprocess_team_graphics (8 base one-hot + 1 battery scalar + 4 item
        # one-hot) plus the 6 MyModel.forward derives (4 unit occupancy + storage free capacity +
        # fetchable battery; see model/derived_obs.py for why each is not learnable from the
        # other channels).
        #
        # Units are drawn straight onto the map grid. An earlier version rasterized them at 4x
        # and folded that back with a stride-4 stem to keep sub-tile position; that is now
        # carried where it is actually used -- on the unit's own token, as a walkability patch
        # plus its offset within its tile (derived_obs.local_wall_features).
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
        # in -> [B, 19, 24, 24] = [B, C, H, W]
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


    def forward(self, graphic):
        """graphic: [B, 19, 24, 24] -- env channels with the derived ones already appended."""
        x = self.pre_conv(graphic)
        x = self.conv64(x)
        x = self.pool(x)
        x = self.conv128(x)

        tokens = self.tokenize_ffn(x.flatten(2).transpose(1, 2))
        return tokens