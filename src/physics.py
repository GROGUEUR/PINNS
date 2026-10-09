"""Résidu de l'EDP par autograd et termes de la perte L_IC, L_BC, L_res (Éq. 1 et 4).

Sprint 2 — auteur A, relecteur B. Dérivation du résidu : docs/physics.md § 4.

Conventions : grandeurs adimensionnées, points en tenseurs (N, 3) de colonnes [x, y, τ],
sorties du réseau en tenseurs (N, 1). Chaque perte renvoie un scalaire (forme ()).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from src.config import CFG, Config
from src.geometry import theta_initial
from src.sampling import CollocationPoints


class LossTerms(NamedTuple):
    """Les trois termes de la perte (Éq. 4), chacun un tenseur scalaire de forme ()."""

    ic: torch.Tensor   # L_IC  : écart à la condition initiale θ₀_ε (Éq. 2)
    bc: torch.Tensor   # L_BC  : écart à θ = 0 sur les murs (Éq. 3)
    res: torch.Tensor  # L_res : résidu de l'EDP aux points intérieurs (Éq. 1)


def pde_residual(model: nn.Module, xyt: torch.Tensor, t_star_max: float) -> torch.Tensor:
    """Résidu de l'équation de la chaleur adimensionnée (Éq. 1, docs/physics.md § 4).

    r = ∂θ/∂τ − t*_max · (∂²θ/∂x² + ∂²θ/∂y²)

    Args:
        model: réseau θ̂(x, y, τ), entrée (N, 3), sortie (N, 1).
        xyt: points de collocation, forme (N, 3), colonnes [x, y, τ], sans dimension.
        t_star_max: horizon adimensionné, facteur issu du changement τ = t*/t*_max.

    Returns:
        Résidu, forme (N, 1). Vaut 0 là où l'EDP est exactement satisfaite.
    """
    # detach() : nouvelle feuille du graphe (même mémoire), le tenseur de l'appelant n'est pas modifié.
    # requires_grad_ : on dérive θ par rapport aux ENTRÉES, pas seulement aux poids.
    xyt = xyt.detach().requires_grad_(True)
    theta = model(xyt)                                                   # (N, 1)
    ones = torch.ones_like(theta)                                        # vecteur v du produit vecteur-jacobien

    # Dérivées premières ; create_graph=True pour pouvoir redériver (ordre 2) puis rétropropager
    grad = torch.autograd.grad(theta, xyt, ones, create_graph=True)[0]  # (N, 3) : [θ_x, θ_y, θ_τ]
    theta_x = grad[:, 0:1]                                               # (N, 1) : « 0:1 » garde la 2e dimension
    theta_y = grad[:, 1:2]                                               # (N, 1)
    theta_tau = grad[:, 2:3]                                             # (N, 1)

    # Dérivées secondes : on redérive θ_x (resp. θ_y) et on ne garde que sa composante en x (resp. y)
    theta_xx = torch.autograd.grad(theta_x, xyt, ones, create_graph=True)[0][:, 0:1]  # (N, 1)
    theta_yy = torch.autograd.grad(theta_y, xyt, ones, create_graph=True)[0][:, 1:2]  # (N, 1)

    return theta_tau - t_star_max * (theta_xx + theta_yy)               # (N, 1)


def loss_ic(model: nn.Module, xyt_ic: torch.Tensor, cfg: Config = CFG) -> torch.Tensor:
    """L_IC : écart quadratique moyen entre θ̂(x, y, 0) et la condition initiale lissée θ₀_ε (Éq. 2).

    Args:
        model: réseau θ̂.
        xyt_ic: points IC, forme (N, 3), colonne τ nulle.
        cfg: configuration (forme, rayon, centre et ε de l'objet, pour θ₀_ε).

    Returns:
        Scalaire (forme ()), sans dimension.
    """
    theta_pred = model(xyt_ic)                                           # (N, 1)
    theta_target = theta_initial(xyt_ic[:, 0:1], xyt_ic[:, 1:2], cfg)    # (N, 1) : même θ₀_ε que le DF
    return torch.mean((theta_pred - theta_target) ** 2)


def loss_bc(model: nn.Module, xyt_bc: torch.Tensor) -> torch.Tensor:
    """L_BC : écart quadratique moyen à θ = 0 sur les murs, c'est-à-dire T = T_amb (Éq. 3).

    Args:
        model: réseau θ̂.
        xyt_bc: points sur les murs, forme (N, 3).

    Returns:
        Scalaire (forme ()), sans dimension.
    """
    theta_pred = model(xyt_bc)            # (N, 1)
    return torch.mean(theta_pred**2)      # la cible vaut 0, donc (θ̂ − 0)² = θ̂²


def loss_res(model: nn.Module, xyt_res: torch.Tensor, t_star_max: float) -> torch.Tensor:
    """L_res : moyenne du carré du résidu de l'EDP aux points intérieurs (Éq. 1).

    Args:
        model: réseau θ̂.
        xyt_res: points intérieurs, forme (N, 3).
        t_star_max: horizon adimensionné (voir pde_residual).

    Returns:
        Scalaire (forme ()), sans dimension.
    """
    residual = pde_residual(model, xyt_res, t_star_max)   # (N, 1)
    return torch.mean(residual**2)


def compute_losses(model: nn.Module, points: CollocationPoints, cfg: Config = CFG) -> LossTerms:
    """Évalue les trois termes de l'Éq. 4, chacun sur sa famille de points.

    Args:
        model: réseau θ̂.
        points: points IC, BC et résidu (src/sampling.py), chacun de forme (N, 3).
        cfg: configuration (objet pour θ₀_ε, T_STAR_MAX pour le résidu).

    Returns:
        LossTerms(ic, bc, res), trois scalaires encore attachés au graphe (rétropropageables).
    """
    return LossTerms(
        ic=loss_ic(model, points.ic, cfg),
        bc=loss_bc(model, points.bc),
        res=loss_res(model, points.res, cfg.T_STAR_MAX),
    )


def total_loss(terms: LossTerms, w_ic: float, w_bc: float, w_res: float) -> torch.Tensor:
    """Perte multi-objectif L = w_IC·L_IC + w_BC·L_BC + w_res·L_res (Éq. 4).

    Les poids sont passés explicitement : cfg.W_* pour la baseline, poids dynamiques
    (src/weighting.py) au sprint 3.

    Args:
        terms: les trois termes de la perte.
        w_ic, w_bc, w_res: poids positifs, sans dimension.

    Returns:
        Scalaire (forme ()), la quantité minimisée par l'optimiseur.
    """
    return w_ic * terms.ic + w_bc * terms.bc + w_res * terms.res
