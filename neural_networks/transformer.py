import torch
import torch.nn as nn
from .decoder import *
from .encoder import *
import numpy as np

# Positionnal encoding
class PositionalEncoding(nn.Module):
    def __init__(self,
        embed_size : int,
        dropout : float = 0.1,
        max_len : int = 100,
        ):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        pe = torch.zeros(1, max_len, embed_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self,
        x : torch.Tensor
        ) -> torch.Tensor:

        # x.shape = (batch_size, seq_len, embedding_dim)

        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)



def create_padding_mask(x : torch.Tensor, padding_idx : int) -> torch.Tensor:
    batch_size, seq_len = x.shape
    ret = (x == padding_idx)
    return torch.nan_to_num(- np.inf * ret, nan = 0.0)


# Definition of a Transformer
class Transformer(nn.Module):

    def __init__(self,
        vocabulary_size_in : int,
        vocabulary_size_out : int,
        pad_fr : int,
        pad_en : int,
        embed_size : int = 128,
        N : int = 4,
        dropout : float = 0.1
        ):

        super(Transformer, self).__init__()

        self.N = N
        self.pad_fr = pad_fr
        self.pad_en = pad_en

        self.positionnal_encoding = PositionalEncoding(embed_size)
        self.emb_in = nn.Embedding(vocabulary_size_in, embed_size, padding_idx = pad_fr)
        self.emb_out = nn.Embedding(vocabulary_size_out, embed_size, padding_idx = pad_en)
        self.encoder = Encoder(embed_size, N, dropout = dropout)
        self.decoder = Decoder(embed_size, N, dropout = dropout)
        self.last_layer = nn.Linear(embed_size, vocabulary_size_out)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self,
        x : torch.Tensor,
        y : torch.Tensor
        ) -> torch.Tensor:

        # Mask for the padding indices
        mask_x = create_padding_mask(x, self.pad_fr)
        mask_y = create_padding_mask(y, self.pad_en)

        # Embedding x and y
        x = self.emb_in(x)
        y = self.emb_out(y)
        
        # Applying positionnal encoding
        x = self.positionnal_encoding(x)
        y = self.positionnal_encoding(y)

        # Applying a dropout to prevent overfitting
        x = self.dropout(x)
        y = self.dropout(y)

        # Encoding the input
        x = self.encoder(x, mask_padding = mask_x)

        # Decoding
        y = self.decoder(x, y, mask_padding_x = mask_x, mask_padding_y = mask_y)

        # Applying a linear layer
        y = self.last_layer(y)

        # Softmax will be done in the cross entropy loss
        # y = nn.functional.softmax(y, dim = -1)

        return y