import math
import torch
import torch.nn as nn
import numpy as np

from attention_is_all_you_need.config import Config as config

class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=config.PAD)

    def forward(self, x):
        for i in range(len(x)):
            if len(x[i]) < config.padding_size:
                x[i].extend([config.UNK] * (config.padding_size - len(x[i])))
            else:
                x[i] = x[i][:config.padding_idx]
        x = self.embedding(torch.tensor(x))
        return x

class Positional_Encoding(nn.Module):
    def __init__(self, d_model):
        super(Positional_Encoding, self).__init__()
        self.d_model = d_model

    def forward(self, seq_len, embedding_dim):
        positional_encoding = np.zeros((seq_len, embedding_dim))
        for pos in range(positional_encoding.shape[0]):
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos / (10000 ** (2 * i / self.d_model)))
        return torch.from_numpy(positional_encoding)