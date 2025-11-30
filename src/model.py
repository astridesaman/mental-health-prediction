import torch
import torch.nn as nn

class MentalHealthModel(nn.Module):
    """
    Réseau de neurones simple pour classification binaire (dépression oui/non)
    Sortie : probabilité entre 0 et 1
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # probabilité
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
