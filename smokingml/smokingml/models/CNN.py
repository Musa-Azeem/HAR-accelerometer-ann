from torch import nn
import torch

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=5, padding='same', bias=False)
        self.ln1 = nn.LayerNorm((2,101))
        self.relu = nn.ReLU()

        self.h1 = nn.Linear(in_features=202, out_features=10)
        self.h2 = nn.Linear(in_features=10, out_features=1)
        

    def forward(self, x):
        x - x.reshape(-1, 3, 101)

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