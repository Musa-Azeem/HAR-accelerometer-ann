from torch import nn
import torch

class CNNModel(nn.Module):
    def __init__(self, in_channels: int, winsize: int):
        super().__init__()
        self.in_channels = in_channels
        self.winsize = winsize

        n_filters = 2
        self.c1 = nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=n_filters, 
            kernel_size=5, 
            padding='same', 
            bias=False
        )
        self.ln1 = nn.LayerNorm((n_filters, self.winsize))
        self.relu = nn.ReLU()

        n_hl = 10
        self.h1 = nn.Linear(in_features=n_filters * self.winsize, out_features=n_hl)
        self.h2 = nn.Linear(in_features=n_hl, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x - x.reshape(-1, self.in_channels, self.winsize)

        x = self.c1(x)
        x = self.ln1(x)
        x = self.relu(x)

        x = x.flatten(start_dim=1)
        x = self.h1(x)
        x = self.relu(x)
        logits = self.h2(x)

        return logits

    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=3e-4)

    @staticmethod
    def get_critersion():
        return nn.BCEWithLogitsLoss()