import math
import torch
import torch.nn as nn
import numpy as np

class Config(object):
    def __init__(self):
        self.vocab_size = 6

        self.d_model = 20
        self.n_heads = 2
        assert self.d_model % self.n_heads == 0

        dim_k = self.d_model % self.n_heads
        dim_v = self.d_model % self.n_heads

        self.padding_size = 30
        self.UNK = 5
        self.PAD = 4
        self.N = 6
        self.p = 0.1

config = Config()