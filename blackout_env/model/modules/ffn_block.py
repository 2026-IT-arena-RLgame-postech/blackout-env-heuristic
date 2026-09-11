# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUBlock(nn.Module):
# (Batch, Length, Input size) -> (Batch, Length, Input size) ??
    def __init__(
            self, input_size: int,
            hidden_size: int,
            output_size: int | None = None
    ) -> None:
        super(SwiGLUBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        if output_size is not None:
            self.output_size = output_size

        self.norm = nn.RMSNorm(input_size)
        self.Wu = nn.Linear(self.input_size, self.hidden_size*2, bias = False)
        self.Wd = nn.Linear(self.hidden_size, self.output_size, bias = False)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        norm_input = self.norm(input)
        x_gate = self.Wu(norm_input)
        x, gate = x_gate.chunk(chunks=2, dim=-1)

        x_gated = F.silu(gate) * x

        return self.Wd(x_gated)
