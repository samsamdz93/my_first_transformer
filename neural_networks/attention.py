import torch
import torch.nn as nn
from numpy import inf

# Function to create a mask
def create_mask(n : int, device : str = 'mps'):
    mask = torch.ones(n,n)
    mask = float(-inf) * mask
    mask = torch.tril(mask, diagonal = -1).transpose(0,1)
    return mask.to(device = device)

# Definition of an Attention layer
class CrossAttention(nn.Module):
    def __init__(self,
        embed_size : int,
        d : int = 64,
        dv : int = None,
        dropout : float = 0.1
        ):

        super(CrossAttention, self).__init__()

        self.d = d
        self.embed_size = embed_size

        # Query and Key matrices
        self.query = nn.Linear(embed_size, d)
        self.key = nn.Linear(embed_size, d)

        # Right and Left factors for the Value matrix
        if dv is None:
            dv = embed_size

        # Decomposition of the value matrix
        self.value = nn.Linear(embed_size, dv)

        self.dropout = nn.Dropout(p = dropout)

    def forward(self,
        x : torch.Tensor,
        y : torch.Tensor = None,
        mask : bool = False,
        mask_padding_x : torch.Tensor = None,
        mask_padding_y : torch.Tensor = None
        ) -> torch.Tensor:

        if y is None:
            y = x
            mask_padding_y = mask_padding_x

        batch_size, seq_len_x, embed_dim = x.shape
        _, seq_len_y, _ = y.shape

        # Query matrix
        Q = self.query(x)
        
        # Key matrix
        K = self.key(y)
        K = torch.transpose(K, 1, 2)

        # Value matrix
        V = self.value(y)
        

        # Computation of the attention score
        QK = torch.matmul(Q, K)/torch.sqrt(torch.tensor(data=self.d, dtype = torch.float))

        # Applying a dropout
        QK = self.dropout(QK)

        # Masking padding tokens
        if mask_padding_y is not None:
            mask_x = mask_padding_x.unsqueeze(-1).repeat(1, 1, seq_len_y)
            mask_y = mask_padding_y.unsqueeze(1).repeat(1, seq_len_x, 1)
            # QK = QK + mask_x # -> add a mask to the bottom
            # QK = QK + mask_y # -> add a mask to the right
            prod_mask = torch.nan_to_num(mask_x * mask_y, posinf = float('inf'))
            final_mask = torch.nan_to_num(prod_mask + mask_y + mask_x, neginf = float('-inf'), posinf = 0.0)
            QK = QK + final_mask

        # Checking if we want a mask
        if mask:
            QK = QK + create_mask(seq_len_x)

        # Applying softmax
        QK_normalized = nn.functional.softmax(QK, dim = -1)

        return torch.matmul(QK_normalized, V)

# Definition of a multihead attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self,
        embed_size : int,
        num_heads : int = 4,
        d : int = 64,
        mask : bool = False,
        dropout : float = 0.1
        ):

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
            self.attentions.append(CrossAttention(embed_size, d = d, dv = self.head_dim, dropout = dropout))
        self.attentions = nn.ModuleList(self.attentions)

    def forward(self,
        x : torch.Tensor,
        y : torch.Tensor = None,
        mask_padding_x : torch.Tensor = None, 
        mask_padding_y  : torch.Tensor = None
        ) -> torch.Tensor:

        batch_size, seq_len, embed_size = x.shape
        outs = [self.attentions[i](x, y, mask = self.mask, mask_padding_x = mask_padding_x, mask_padding_y = mask_padding_y) for i in range(self.num_heads)]
        outs = torch.cat(outs, dim = -1)
        return self.last_linear(outs)





