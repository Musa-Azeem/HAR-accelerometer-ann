import torch
from torch import nn

WIN_SIZE = 500
n_hl = 10

class MLP_10hl_500ws(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(WIN_SIZE*3, n_hl),
            nn.ReLU(),
            nn.Linear(n_hl, 1)
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits 