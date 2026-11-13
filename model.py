import torch
from torch import nn


class BikeNet(nn.Module):
    """Neural network for predicting bike rentals."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 16,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
