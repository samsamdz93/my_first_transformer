import torch
import torch.nn as nn
from .decoder import *
from .encoder import *
import numpy as np

# Positionnal encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout = 0.1, max_len = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, 1, embed_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0, 1))


# Definition of a Transformer
class Transformer(nn.Module):
    def __init__(self, vocabulary_size_in, vocabulary_size_out, embed_size = 128, N = 4):
        super(Transformer, self).__init__()
        self.N = N
        self.positionnal_encoding = PositionalEncoding(embed_size)
        self.emb_in = nn.Embedding(vocabulary_size_in, embed_size)
        self.emb_out = nn.Embedding(vocabulary_size_out, embed_size)
        self.encoder = Encoder(embed_size, N)
        self.decoder = Decoder(embed_size, N)
        self.last_layer = nn.Linear(embed_size, vocabulary_size_out)

    def forward(self, x, y):
        # Embedding the inputs
        x = self.emb_in(x)
        x = x + self.positionnal_encoding(x)
        y = self.emb_out(y)
        y = y + self.positionnal_encoding(y)

        # Encoding the input
        x = self.encoder(x)

        # Decoding
        y = self.decoder(x, y)

        y = self.last_layer(y)
        # y = nn.functional.softmax(y, dim = -1)
        # Softmax will be done in the cross entropy loss
        return y