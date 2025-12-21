import torch
import torch.nn as nn

class LinearBaseline(nn.Module):
    """
    Modèle linéaire de référence (régression logistique).
    Une seule couche linéaire suivie d'une sigmoïde.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, input_dim)
        logits = self.linear(x)          # (batch_size, 1)
        probs  = torch.sigmoid(logits)   # (batch_size, 1) dans [0,1]
        return probs


class MentalHealthModelNN(nn.Module):
    """
    Réseau de neurones (MLP) pour classification binaire.
    Sortie : probabilité entre 0 et 1 (via Sigmoid).
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
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applique toutes les couches définies dans self.net
        return self.net(x)

class MentalHealthModelNNv2(nn.Module):
    """
    Version plus profonde du MLP.
    Testée pour augmenter la capacité du modèle.
    """
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)