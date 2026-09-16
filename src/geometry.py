"""Objet chaud (disque ou pavé) et condition initiale lissée θ₀_ε (Éq. 2 du sujet).

Sprint 1 — auteur A, relecteur B. Dérivation complète dans docs/physics.md § 5.

Toutes les fonctions sont élément par élément : elles acceptent des tenseurs de
n'importe quelle forme (colonne (N, 1) pour le PINN, grille (FD_N, FD_N) pour le DF)
et renvoient la même forme. Les coordonnées sont adimensionnées, dans [0, 1].
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import CFG, Config


def distance_to_center(x: torch.Tensor, y: torch.Tensor, cfg: Config = CFG) -> torch.Tensor:
    """Distance d(x, y) au centre de l'objet, dans la métrique propre à sa forme.

    Disque : distance euclidienne √((x−cx)² + (y−cy)²), le bord est le cercle d = R.
    Pavé   : distance de Chebyshev max(|x−cx|, |y−cy|), le bord est le carré d = R
             (R est alors le demi-côté).

    Args:
        x, y: coordonnées sans dimension, même forme, typiquement (N, 1).
        cfg: configuration (forme, centre cx, cy).

    Returns:
        Distance, même forme que x, sans dimension.
    """
    dx = x - cfg.OBJECT_CX
    dy = y - cfg.OBJECT_CY
    if cfg.OBJECT_SHAPE == "disque":
        return torch.sqrt(dx**2 + dy**2)
    return torch.maximum(dx.abs(), dy.abs())  # "pave" (les autres formes sont refusées par Config)


def theta_initial(x: torch.Tensor, y: torch.Tensor, cfg: Config = CFG) -> torch.Tensor:
    """Condition initiale lissée θ₀_ε(x, y) = ½ [1 − tanh((d − R) / ε)]  (Éq. 2, adimensionnée).

    Vaut ≈ 1 dans l'objet, ≈ 0 dehors, exactement ½ sur le bord. La transition a une
    largeur ≈ 2ε : un échelon n'est pas représentable par un MLP lisse, et le DF de
    référence utilise le même θ₀_ε pour que la comparaison soit équitable.

    Args:
        x, y: coordonnées sans dimension, même forme, typiquement (N, 1).
        cfg: configuration (forme, rayon R, centre, largeur ε = EPS_IC).

    Returns:
        θ₀_ε ∈ [0, 1], même forme que x, sans dimension.
    """
    d = distance_to_center(x, y, cfg)
    return 0.5 * (1.0 - torch.tanh((d - cfg.OBJECT_RADIUS) / cfg.EPS_IC))


def initial_condition_grid(cfg: Config = CFG) -> np.ndarray:
    """θ₀_ε évaluée sur la grille du solveur DF, pour partager la même IC que le PINN.

    Convention d'indexation : grid[i, j] = θ₀_ε(x_i, y_j) avec x_i = i·dx, y_j = j·dx
    (meshgrid en mode « ij »). Le solveur DF et les métriques doivent la respecter.

    Args:
        cfg: configuration (FD_N nœuds par direction, paramètres de l'objet).

    Returns:
        Tableau NumPy float64 de forme (FD_N, FD_N), valeurs dans [0, 1].
    """
    coords = torch.linspace(0.0, 1.0, cfg.FD_N, dtype=torch.float64)   # (FD_N,)
    x, y = torch.meshgrid(coords, coords, indexing="ij")                # (FD_N, FD_N) chacun
    return theta_initial(x, y, cfg).numpy()


def theta_to_celsius(theta: torch.Tensor, cfg: Config = CFG) -> torch.Tensor:
    """Retour aux unités physiques : T = T_amb + (T_obj − T_amb) · θ, en °C."""
    return cfg.T_AMB + cfg.delta_t * theta


def celsius_to_theta(temp: torch.Tensor, cfg: Config = CFG) -> torch.Tensor:
    """Adimensionnement de la température : θ = (T − T_amb) / (T_obj − T_amb)."""
    return (temp - cfg.T_AMB) / cfg.delta_t
