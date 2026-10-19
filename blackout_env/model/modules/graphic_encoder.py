from torch import nn

from blackout_env.model.derived_obs import N_DERIVED_MAP_CHANNELS

from .bottleneck_block import BottleNeckBlock
from .ffn_block import SwiGLUBlock


# map_encoder -> (kernel, stride, padding) of the ViT-style linear patch embedding.  Both give 6x6 = 36
# tokens from the 24x24 grid in the CNN's row-major order (so the trunk's RoPE positions are unchanged):
# patch4 tiles the grid; patch6 overlaps neighbouring patches by 2 cells (padding 1 on each side).
PATCH_ENCODERS = {"patch4": (4, 4, 0), "patch6": (6, 4, 1)}
MAP_ENCODERS = ("cnn", *PATCH_ENCODERS)


class GraphicEncoder(nn.Module):
    def __init__(self, hidden_size : int = 256, in_channels : int = 13 + N_DERIVED_MAP_CHANNELS,
                 map_encoder : str = "cnn") -> None:
        super(GraphicEncoder, self).__init__()

        if map_encoder not in MAP_ENCODERS:
            raise ValueError(f"map_encoder must be one of {MAP_ENCODERS}, got {map_encoder!r}")
        self.hidden_size = hidden_size
        self.map_encoder = map_encoder
        if map_encoder != "cnn":
            # ViT-style tokenizer (A/B in docs/rl_finetune_plan_ko.md): one linear patch embedding and a
            # LayerNorm; no convolutional stages, no feature maps.  Only the "cnn" path creates the CNN
            # modules, so a default net's parameters and state_dict keys are exactly what they were.
            k, s, p = PATCH_ENCODERS[map_encoder]
            self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=k, stride=s, padding=p)
            self.patch_norm = nn.LayerNorm(hidden_size)
            return

        # graphic : [24, 24, 19] = [H, W, C] -- 13 channels from
        # MyObsPreprocessor.preprocess_team_graphics (8 base one-hot + 1 battery scalar + 4 item
        # one-hot) plus the 6 MyModel.forward derives (4 unit occupancy + storage free capacity +
        # fetchable battery; see model/derived_obs.py for why each is not learnable from the
        # other channels).
        #
        # Units are drawn straight onto the map grid. An earlier version rasterized them at 4x
        # and folded that back with a stride-4 stem to keep sub-tile position; that is now
        # carried where it is actually used -- on the unit's own token, as walkability sampled
        # around its true position (derived_obs.local_wall_features).
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


    def forward(self, graphic, features: bool = False):
        """graphic: [B, 19, 24, 24] -- env channels with the derived ones already appended.
        ``features``: also return the feature maps ``{"f2": [B, 64, 12, 12], "f3": [B, 128, 6, 6]}``
        (MyModel's map-token cross-attention reads them)."""
        if self.map_encoder != "cnn":
            if features:
                raise ValueError("feature maps (map-token cross-attention) exist only for map_encoder='cnn'")
            x = self.patch_embed(graphic)                            # [B, hidden, 6, 6]
            return self.patch_norm(x.flatten(2).transpose(1, 2))     # [B, 36, hidden]
        x = self.pre_conv(graphic)
        f2 = self.conv64(x)
        x = self.pool(f2)
        x = self.conv128(x)

        tokens = self.tokenize_ffn(x.flatten(2).transpose(1, 2))
        if features:
            return tokens, {"f2": f2, "f3": x}
        return tokens