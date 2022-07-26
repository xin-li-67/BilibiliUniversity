import math
import torch
import torch.nn as nn
import numpy as np

from attention_is_all_you_need.decoder import Decoder
from attention_is_all_you_need.embedding import Embedding
from attention_is_all_you_need.encoder import Encoder
from attention_is_all_you_need.config import Config as config

class Transformer_Layer(nn.Module):
    def __init__(self):
        super(Transformer_Layer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x_input, x_output = x
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_output, encoder_output)
        return (encoder_output, decoder_output)

class Transformer(nn.Module):
    def __init__(self, N, vocab_size, output_dim):
        super(Transformer, self).__init__()
        self.embedding_input = Embedding(vocab_size=vocab_size)
        self.embedding_output = Embedding(vocab_size=vocab_size)

        self.output_dim = output_dim
        self.linear = nn.Linear(config.d_model, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.model = nn.Sequential(*[Transformer_Layer() for _ in range(N)])

    def forward(self, x):
        x_input, x_output = x
        x_input = self.embedding_input(x_input)
        x_output = self.embedding_output(x_output)

        _, output = self.model((x_input, x_output))

        output = self.linear(output)
        output = self.softmax(output)
        return output