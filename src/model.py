"""Réseau θ̂(x, y, τ) : MLP tanh (sprint 2).

Sprint 2 — auteur A, relecteur B.
Restent à ajouter par A : la variante hard constraints (sprint 3),
θ̂ = θ₀_ε(x, y) + τ · x(1−x) · y(1−y) · N(x, y, τ), puis la variante paramétrique (sprint 4).
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import CFG, Config


def to_symmetric_range(xyt: torch.Tensor) -> torch.Tensor:
    """Remet les entrées de [0, 1] dans [−1, 1] : z = 2·xyt − 1.

    tanh est centrée en 0 et quasi linéaire autour de 0 : des entrées centrées évitent
    de démarrer dans la zone saturée de tanh, où les gradients s'écrasent.

    Args:
        xyt: points, forme (N, 3), colonnes [x, y, τ] dans [0, 1], sans dimension.

    Returns:
        Tenseur (N, 3) dans [−1, 1].
    """
    return 2.0 * xyt - 1.0


class MLP(nn.Module):
    """Perceptron multicouche θ̂(x, y, τ) : 3 → hidden_layers → 1, activation tanh.

    - tanh plutôt que ReLU : le résidu contient des dérivées secondes, or celle de ReLU
      est nulle presque partout, donc le laplacien du réseau serait nul.
    - Dernière couche linéaire, sans activation : la sortie θ n'est pas bornée par
      construction, c'est la perte qui la ramène dans [0, 1].

    Args:
        hidden_layers: largeurs des couches cachées, (64, 64, 64, 64) par défaut.
    """

    def __init__(self, hidden_layers: tuple[int, ...] = CFG.HIDDEN_LAYERS) -> None:
        super().__init__()
        sizes = (3, *hidden_layers, 1)  # ex. (3, 64, 64, 64, 64, 1) : entrées (x, y, τ), sortie θ
        layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(n_in, n_out))
        self.layers = nn.ModuleList(layers)  # ModuleList : PyTorch enregistre les poids de chaque couche
        self.activation = nn.Tanh()
        self._init_xavier()

    def _init_xavier(self) -> None:
        """Init Xavier (Glorot) normale, biais nuls (AGENTS § 6, comme Raissi et al. 2019).

        Variance 2 / (n_in + n_out) : l'amplitude du signal reste du même ordre d'une
        couche à l'autre, ni écrasée ni amplifiée, ce qui garde tanh hors saturation.
        """
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """θ̂ aux points xyt.

        Args:
            xyt: points, forme (N, 3), colonnes [x, y, τ] dans [0, 1], sans dimension.

        Returns:
            θ̂, forme (N, 1), sans dimension (θ = 0 à T_amb, θ = 1 à T_obj).
        """
        h = to_symmetric_range(xyt)            # (N, 3) dans [−1, 1]
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))      # (N, largeur) : couche cachée = linéaire puis tanh
        return self.layers[-1](h)              # (N, 1) : sortie linéaire


def build_model(cfg: Config = CFG) -> nn.Module:
    """Construit le réseau décrit par la config et le place sur cfg.DEVICE.

    Point d'entrée unique, utilisé par l'entraînement et par le rechargement d'un
    checkpoint : les deux construisent ainsi exactement la même architecture.

    Args:
        cfg: configuration (HIDDEN_LAYERS, DEVICE).

    Returns:
        Réseau θ̂(x, y, τ), sur cfg.DEVICE.
    """
    return MLP(cfg.HIDDEN_LAYERS).to(cfg.DEVICE)
