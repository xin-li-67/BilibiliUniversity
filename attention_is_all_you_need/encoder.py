import math
import torch
import torch.nn as nn
import numpy as np

from attention_is_all_you_need.embedding import Positional_Encoding
from attention_is_all_you_need.config import Config as config

class Multihead_Attention(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads):
        super(Multihead_Attention, self).__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)

        self.o = nn.Linear(dim_v, d_model)
        self.norm_factor = 1 / math.sqrt(d_model)

    def generate_mask(self, dim):
        # sequence mask
        matrix = np.ones((dim, dim))
        mask = torch.Tensor(np.tril(matrix))
        return mask == 1

    def forward(self, x, y, require_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0

        # self attention
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        V = self.v(x).reshape(-1, y.shape[0], y.shape[1], self.dim_v // self.n_heads)
        print(f'Attention V shape is {V.shape}')

        attention_score = torch.matmul(Q, K.permuate(0, 1, 3, 2)) * self.norm_factor
        if require_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score.masked_fill(mask, value=float('-inf'))
        output = torch.matmul(attention_score, V).reshape(y.shape[0], y.shape[1], -1)
        print(f'Attention output shape is {output.shape}')

        output = self.o(output)
        return output

class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        output = nn.ReLU()(self.l1(x))
        output = self.l2(output)
        return output

class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(config.p)

    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        x = self.dropout(x + sub_output)

        layer_norm = nn.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.multi_atten = Multihead_Attention(config.d_model, config.dim_k, config.dim_v, config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)
        self.add_norm = Add_Norm()

    def forward(self, x):
        x += self.positional_encoding(x.shape[1], config.d_model)
        output = self.add_norm(x, self.multi_atten, y=x)
        output = self.add_norm(output, self.feed_forward)
        return output