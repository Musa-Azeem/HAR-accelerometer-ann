import torch
from torch import nn

class MLP_1hl(nn.Module):
    def __init__(self, n_hl: int, n_features: int) -> None:
        """
            Initialize MLP with n_hl hidden layer neurons and n_features
            input features
        Args:
            n_hl (int): number of hidden layer neurons
            n_features (int): number of input features (window size of dataset*3)
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_features, n_hl),
            nn.ReLU(),
            nn.Linear(n_hl, 1)
        )
    
    def forward(self, x: torch.tensor) -> None:
        """
            Forward pass on MLP. Returns logits without Sigmoid

        Args:
            x (torch.tensor): tensor of length 1500
        """
        logits = self.linear_relu_stack(x)
        return logits

    @staticmethod
    def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
        """
            Returns Adam optimizer to be used with this model

        Args:
            model (torch.nn): instantiated model

        Returns:
            torch.optim.Optimizer: Adam optimizer
        """
        return torch.optim.Adam(model.parameters(), lr=0.001)

    @staticmethod
    def get_criterion() -> nn.Module:
        """
            Return BCE with logits loss function

        Returns:
            nn.Module: BCEWithLogitsLoss function
        """
        return nn.BCEWithLogitsLoss()