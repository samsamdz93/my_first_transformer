import torch
import torch.nn as nn
from .attention import *
from .mlp import *





# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, dropout = 0.1):
        super(EncoderBlock, self).__init__()
        self.att = MultiHeadAttention(embed_size, dropout = dropout)
        self.mlp = MLP([embed_size, 2 * embed_size, embed_size])
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self,
        x : torch.Tensor,
        mask_padding : torch.Tensor,
        ) -> torch.Tensor:

        batch_size, sequence_length, embed_dim = x.shape
        x = nn.functional.layer_norm(x + self.dropout(self.att(x, mask_padding_x = mask_padding)), [sequence_length, embed_dim])
        x = nn.functional.layer_norm(x + self.dropout(self.mlp(x)), [sequence_length, embed_dim])
        return x

# Definition of an Encoder
class Encoder(nn.Module):
    def __init__(self, embed_size : int, N : int, dropout : float = 0.1):
        super(Encoder, self).__init__()
        self.blocks = []
        self.N = N
        for _ in range(N):
            self.blocks.append(EncoderBlock(embed_size, dropout = dropout))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self,
        x : torch.Tensor,
        mask_padding : torch.Tensor = None,
        ) -> torch.Tensor:

        for i in range(self.N):
            x = self.blocks[i](x, mask_padding = mask_padding)
        return x




