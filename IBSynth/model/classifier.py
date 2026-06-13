from torch import nn, Tensor
from typing import Sequence


class BinaryClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
