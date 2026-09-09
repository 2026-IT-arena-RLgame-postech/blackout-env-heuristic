from torch import nn


class VectorEncoder(nn.Module):
    def __init__(self, hidden_size : int = 256):
        super(VectorEncoder, self).__init__()

    def forward(self, input_tensor):
        pass