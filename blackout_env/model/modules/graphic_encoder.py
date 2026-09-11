from torch import nn

from .bottleneck_block import BottleNeckBlock
from .ffn_block import SwiGLUBlock


class GraphicEncoder(nn.Module):
    def __init__(self, hidden_size : int = 256, in_channels : int = 13) -> None:
        super(GraphicEncoder, self).__init__()

        self.hidden_size = hidden_size

        # graphic : [24, 24, 13] = [H, W, C] from MyObsPreprocessor.preprocess_team_graphics
        # (8 base one-hot + 1 battery scalar + 4 item one-hot; no unit channels — units are
        # carried by agent_states instead)
        # to be flipped by setting tensor format
        # in -> [B, 24, 24, 13] = [B, H, W, C]
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2), # -> [B, 64, 24, 24]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> [B, 128, 12, 12]
        )

        self.conv64 = nn.Sequential(  # -> [B, 128, 12, 12]
            BottleNeckBlock(128, 32, 16, 4),
            BottleNeckBlock(128, 32, 16, 4),
        )
        self.pool = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # -> [B, 256, 6, 6]

        self.conv128 = nn.Sequential( # -> [B, 256, 6, 6]
            BottleNeckBlock(256, 64, 16, 4),
            BottleNeckBlock(256, 64, 16, 4),
            BottleNeckBlock(256, 64, 16, 4),
        )
        # -> [B, 256, 36]

        self.tokenize_ffn = SwiGLUBlock(256, 1024, self.hidden_size)


    def forward(self, graphic):
        x = self.pre_conv(graphic)
        x = self.conv64(x)
        x = self.pool(x)
        x = self.conv128(x)

        tokens = self.tokenize_ffn(x.flatten(2).transpose(1, 2))
        return tokens