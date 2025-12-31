import torch
import torch.nn as nn
from .attention import *
from .mlp import *
# Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size : int):
        super(DecoderBlock, self).__init__()
        self.att1 = MultiHeadAttention(embed_size, mask = True)
        self.att2 = MultiHeadAttention(embed_size)
        self.mlp = MLP([embed_size, 2 * embed_size, embed_size])
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x, y, mask_padding_x, mask_padding_y): # x is the context (encoder's output) and y is the input of the decoder
        batch_size, sequence_length, embed_dim = y.shape
        y = nn.functional.layer_norm(y + self.dropout(self.att1(y, mask_padding_x = mask_padding_y)), [sequence_length, embed_dim])
        y = nn.functional.layer_norm(y + self.dropout(self.att2(y, y = x, mask_padding_x = mask_padding_y, mask_padding_y = mask_padding_x)), [sequence_length, embed_dim])
        y = nn.functional.layer_norm(y + self.dropout(self.mlp(y)), [sequence_length, embed_dim])
        return y

# Definition of an Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size : int, N : int):

        super(Decoder, self).__init__()

        self.blocks = []
        self.N = N

        for _ in range(N):
            self.blocks.append(DecoderBlock(embed_size))

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, 
        x : torch.Tensor, # Context
        y : torch.Tensor, # Input of the decoder
        mask_padding_x : torch.Tensor = None,
        mask_padding_y : torch.Tensor = None
        ) -> torch.Tensor: 
    
        for i in range(self.N):
            y = self.blocks[i](x, y, mask_padding_x = mask_padding_x, mask_padding_y = mask_padding_y)
        return y


