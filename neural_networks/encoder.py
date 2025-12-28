import torch
import torch.nn as nn
from .attention import *
from .mlp import *



# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, embed_size):
        super(EncoderBlock, self).__init__()
        self.att = MultiHeadAttention(embed_size)
        self.mlp = MLP([embed_size, 2 * embed_size, embed_size])

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape
        x = nn.functional.layer_norm(x + self.att(x), [sequence_length, embed_dim])
        x = nn.functional.layer_norm(x + self.mlp(x), [sequence_length, embed_dim])
        return x

# Definition of an Encoder
class Encoder(nn.Module):
    def __init__(self, embed_size, N):
        super(Encoder, self).__init__()
        self.blocks = []
        self.N = N
        for _ in range(N):
            self.blocks.append(EncoderBlock(embed_size))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        for i in range(self.N):
            x = self.blocks[i](x)
        return x




