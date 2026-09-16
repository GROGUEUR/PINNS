"""Points de collocation : IC, BC et résidu (travail 1 du sujet).

Sprint 1 — auteur A, relecteur B. Le RAD (sprint 3, auteur B) réutilisera
``sample_residual_points(cfg, n=cfg.RAD_POOL_SIZE)`` pour tirer son pool de candidats.

Conventions :
- chaque famille est un tenseur (N, 3) de colonnes [x, y, τ], toutes dans [0, 1] ;
- les tenseurs sont créés sur ``cfg.DEVICE``, là où vit le modèle ;
- la reproductibilité vient de ``set_seeds()`` (générateur global de torch).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from src.config import CFG, Config
from src.geometry import distance_to_center


class CollocationPoints(NamedTuple):
    """Les trois familles de points de la perte (Éq. 4), chacune de forme (N, 3)."""

    ic: torch.Tensor   # (N_IC, 3), τ = 0        → L_IC
    bc: torch.Tensor   # (N_BC, 3), sur les murs → L_BC
    res: torch.Tensor  # (N_RES, 3), intérieur   → L_res


def sample_tau(n: int, cfg: Config = CFG) -> torch.Tensor:
    """Instants τ ∈ [0, 1] concentrés près de 0 : τ = u^TAU_EXPONENT avec u ~ U[0, 1].

    Avec l'exposant 2, P(τ < 0.05) = √0.05 ≈ 22 % au lieu de 5 % : on suit la dynamique,
    qui se joue dans les premiers % du temps (θ_max passe de 1 à 0.39 pour τ < 0.05).

    Args:
        n: nombre d'instants à tirer.
        cfg: configuration (TAU_EXPONENT, DEVICE).

    Returns:
        Tenseur (n, 1), sans dimension.
    """
    u = torch.rand(n, 1, device=cfg.DEVICE)  # (n, 1)
    return u**cfg.TAU_EXPONENT


def _sample_near_object_edge(n: int, cfg: Config) -> torch.Tensor:
    """n points (x, y) uniformes dans la bande |d − R| ≤ ic_edge_band, tirés par rejet.

    On tire des candidats uniformes dans la pièce et on ne garde que ceux proches du
    bord, jusqu'à en avoir n. La méthode vaut pour toute forme d'objet (pas de formule
    polaire) et reste vectorisée : la boucle porte sur des lots, pas sur des points.

    Returns:
        Tenseur (n, 2).
    """
    kept = torch.empty(0, 2, device=cfg.DEVICE)                                  # (0, 2)
    while kept.shape[0] < n:
        xy = torch.rand(n, 2, device=cfg.DEVICE)                                 # (n, 2) candidats
        d = distance_to_center(xy[:, 0:1], xy[:, 1:2], cfg)                      # (n, 1)
        in_band = ((d - cfg.OBJECT_RADIUS).abs() <= cfg.ic_edge_band).squeeze(1)  # (n,) booléen
        kept = torch.cat([kept, xy[in_band]], dim=0)
    return kept[:n]


def sample_ic_points(cfg: Config = CFG, n: int | None = None) -> torch.Tensor:
    """Points de la condition initiale (x, y, τ = 0), densifiés près du bord de l'objet.

    Une part IC_EDGE_FRACTION des points est tirée dans la bande autour du bord, où θ₀_ε
    passe de 1 à 0 sur une largeur ≈ 2ε ; le reste est uniforme dans la pièce, pour que
    L_IC contrôle aussi l'intérieur de l'objet et le fond à θ = 0.

    Args:
        cfg: configuration (N_IC, IC_EDGE_FRACTION, objet).
        n: nombre de points ; N_IC par défaut.

    Returns:
        Tenseur (n, 3), colonnes [x, y, 0].
    """
    n = cfg.N_IC if n is None else n
    n_edge = round(cfg.IC_EDGE_FRACTION * n)
    xy_edge = _sample_near_object_edge(n_edge, cfg)              # (n_edge, 2)
    xy_uniform = torch.rand(n - n_edge, 2, device=cfg.DEVICE)    # (n − n_edge, 2)
    xy = torch.cat([xy_edge, xy_uniform], dim=0)                 # (n, 2)
    tau = torch.zeros(n, 1, device=cfg.DEVICE)                   # (n, 1) : τ = 0 (Éq. 2)
    return torch.cat([xy, tau], dim=1)


def sample_bc_points(cfg: Config = CFG, n: int | None = None) -> torch.Tensor:
    """Points des conditions aux limites : n/4 points par mur, τ uniforme sur [0, 1].

    τ n'est pas biaisé ici : la BC θ = 0 (Éq. 3) doit tenir à tout instant et n'est pas
    plus difficile près de τ = 0, où θ vaut déjà ≈ 0 loin de l'objet.

    Args:
        cfg: configuration (N_BC, DEVICE).
        n: nombre de points, multiple de 4 ; N_BC par défaut.

    Returns:
        Tenseur (n, 3), chaque ligne vérifiant x ∈ {0, 1} ou y ∈ {0, 1}.
    """
    n = cfg.N_BC if n is None else n
    if n % 4 != 0:
        raise ValueError(f"n = {n} doit être un multiple de 4 (un quart par mur)")
    n_wall = n // 4
    s = [torch.rand(n_wall, 1, device=cfg.DEVICE) for _ in range(4)]  # abscisse le long de chaque mur
    zeros, ones = torch.zeros_like(s[0]), torch.ones_like(s[0])
    xy = torch.cat(
        [
            torch.cat([zeros, s[0]], dim=1),  # mur x = 0
            torch.cat([ones, s[1]], dim=1),   # mur x = 1
            torch.cat([s[2], zeros], dim=1),  # mur y = 0
            torch.cat([s[3], ones], dim=1),   # mur y = 1
        ],
        dim=0,
    )                                                            # (n, 2)
    tau = torch.rand(n, 1, device=cfg.DEVICE)                    # (n, 1)
    return torch.cat([xy, tau], dim=1)


def sample_residual_points(cfg: Config = CFG, n: int | None = None) -> torch.Tensor:
    """Points intérieurs (x, y, τ) où l'on évalue le résidu de l'EDP (Éq. 1).

    x et y sont uniformes dans la pièce ; τ est biaisé vers 0 par ``sample_tau``.

    Args:
        cfg: configuration (N_RES, TAU_EXPONENT, DEVICE).
        n: nombre de points ; N_RES par défaut.

    Returns:
        Tenseur (n, 3), colonnes [x, y, τ].
    """
    n = cfg.N_RES if n is None else n
    xy = torch.rand(n, 2, device=cfg.DEVICE)   # (n, 2)
    tau = sample_tau(n, cfg)                   # (n, 1)
    return torch.cat([xy, tau], dim=1)


def sample_collocation_points(cfg: Config = CFG) -> CollocationPoints:
    """Tire les trois familles de points en une fois (début d'entraînement, ou L-BFGS figé)."""
    return CollocationPoints(
        ic=sample_ic_points(cfg),
        bc=sample_bc_points(cfg),
        res=sample_residual_points(cfg),
    )
