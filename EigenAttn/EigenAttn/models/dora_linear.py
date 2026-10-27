import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.rank = rank
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, in_features))
        nn.init.zeros_(self.base.weight)
        self.base.weight.requires_grad = False

    def forward(self, x):
        dora_update = self.A @ self.B
        return F.linear(x, self.base.weight + dora_update)

