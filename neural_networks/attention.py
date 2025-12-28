import torch
import torch.nn as nn
from numpy import inf

# Function to create a mask
def create_mask(n, device = 'mps'):
    mask = torch.ones(n,n)
    mask = float(-inf) * mask
    mask = torch.tril(mask, diagonal = -1).transpose(0,1)
    return mask.to(device = device)


# Definition of an Attention layer
class CrossAttention(nn.Module):
    def __init__(self, embed_size, d = 64, dv = None):
        super(CrossAttention, self).__init__()
        self.d = d
        self.embed_size = embed_size

        # Query and Key matrices
        self.query = nn.Linear(embed_size, d)
        self.key = nn.Linear(embed_size, d)

        # Right and Left factors for the Value matrix
        if dv is None:
            dv = embed_size
        self.valueR = nn.Linear(embed_size, d)
        self.valueL = nn.Linear(d, dv)

    def forward(self, x, y = None, mask = False):
        batch_size, sequence_length, embed_dim = x.shape
        Q = self.query(x)
        if y is None:
            y = x
        K = self.key(y)
        K = torch.transpose(K, 1, 2)

        # Computation of the attention score
        QK = torch.matmul(Q, K)/torch.sqrt(torch.tensor(data=self.d, dtype = torch.float))

        # Checking if we want a mask
        if mask:
            QK = QK + create_mask(sequence_length)
        QK_normalized = nn.functional.softmax(QK, dim = 1)

        # Value matrix
        V = self.valueL(self.valueR(y))
        return torch.matmul(QK_normalized, V)

# Definition of a multihead attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads = 4, d = 64, mask = False):
        super(MultiHeadAttention, self).__init__()
        self.d = d
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.mask = mask

        # Checking if embedding size is divisible by number of heads
        if embed_size % num_heads != 0:
            raise ValueError('Embedding size must be divisible by the number of heads.')

        self.head_dim = embed_size // num_heads
        self.last_linear = nn.Linear(embed_size, embed_size)

        # Different keys and queries matrices
        self.attentions = []
        for _ in range(num_heads):
            self.attentions.append(CrossAttention(embed_size, d = d, dv = self.head_dim))
        self.attentions = nn.ModuleList(self.attentions)

    def forward(self, x, y = None):
        batch_size, seq_len, embed_size = x.shape
        outs = [self.attentions[i](x, y, self.mask) for i in range(self.num_heads)]
        outs = torch.cat(outs, dim = -1)
        return self.last_linear(outs)
