import math
import torch
import torch.nn as nn
import numpy as np

from attention_is_all_you_need.config import Config as config
from attention_is_all_you_need.embedding import Positional_Encoding
from attention_is_all_you_need.encoder import Add_Norm, Feed_Forward, Multihead_Attention

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.multi_atten = Multihead_Attention(config.d_model, config.dim_k, config.dim_v, config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)
        self.add_norm = Add_Norm()

    def forward(self, x, encoder_output):
        x += self.positional_encoding(x.shape[1], config.d_model)
        output = self.add_norm(x, self.multi_atten, y=x, require_mask=True)
        output = self.add_norm(output, self.multi_atten, y=encoder_output, require_mask=True)
        output = self.add_norm(output, self.feed_forward)
        return output