import torch
from torch import nn
from modules import *

class MyModel(nn.Module):
    def __init__(self, hidden_size : int = 256):
        super(MyModel, self).__init__()

        self.hidden_size = hidden_size

        # encoder
        self.graphic_encoder = GraphicEncoder(self.hidden_size)
        self.vector_encoder = VectorEncoder()

        # main trunk
        # [36 vision] , [5 ally], [5 enemy], [scores, time]
        self.attention = AttentionLayers(self.hidden_size, 8, 12)

        # vision token projection
        self.spr_head = SwiGLUBlock(self.hidden_size, self.hidden_size*3, self.hidden_size)

        self.q_head = nn.Sequential(
            SwiGLUBlock(self.hidden_size, self.hidden_size*3, self.hidden_size),
            nn.Linear(self.hidden_size, 8),
        )



    def forward(self, graphic : torch.Tensor, vector : torch.Tensor):
        vis_in = self.graphic_encoder(graphic)
        vec_in = self.vector_encoder(vector)

        # need to be checked
        tokens_in = torch.cat((vis_in, vec_in), dim = 1)
        tokens_out = self.attention(tokens_in)

        # need to be changed
        vis_out  = tokens_out[:,0,:]
        ally_out = tokens_out[:,1:,:]
        enemy_out = tokens_out[:,:-1,:]
        stat_out  = tokens_out[:,-1,:]

        vision_latent_out = self.spr_head(vis_out)
        q_out = self.q_head(ally_out)

