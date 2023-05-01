import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, n_hl, win_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(win_size*3, n_hl),
            nn.ReLU(),
            nn.Linear(n_hl, 1)
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=0.001)

    @staticmethod
    def get_criterion():
        return nn.BCEWithLogitsLoss()