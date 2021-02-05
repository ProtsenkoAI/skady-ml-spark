import torch
from torch import nn

class MFWithBiasModel(nn.Module):
    def __init__(self, nusers, nitems, hidden_size):
        self.nusers = nusers
        self.nitems = nitems
        self.hidden_size = hidden_size
        super().__init__()
