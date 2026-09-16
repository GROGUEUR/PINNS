"""Tests de src/geometry.py : distance à l'objet, IC lissée θ₀_ε, conversions θ ↔ T."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.config import CFG
from src.geometry import (
    celsius_to_theta,
    distance_to_center,
    initial_condition_grid,
    theta_initial,
    theta_to_celsius,
)

DISQUE = replace(CFG, OBJECT_SHAPE="disque", OBJECT_RADIUS=0.1, OBJECT_CX=0.5, OBJECT_CY=0.5)
PAVE = replace(CFG, OBJECT_SHAPE="pave", OBJECT_RADIUS=0.1, OBJECT_CX=0.5, OBJECT_CY=0.5)


def _point(x: float, y: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Un point (x, y) sous forme de deux tenseurs (1, 1), convention du dépôt."""
    return torch.tensor([[x]]), torch.tensor([[y]])


def test_distance_disque_est_euclidienne() -> None:
    x, y = _point(0.5 + 0.3, 0.5 + 0.4)  # triangle 3-4-5 : d = 0.5
    assert distance_to_center(x, y, DISQUE).item() == pytest.approx(0.5)


def test_distance_pave_est_chebyshev() -> None:
    x, y = _point(0.5 + 0.3, 0.5 + 0.4)  # max(0.3, 0.4) = 0.4
    assert distance_to_center(x, y, PAVE).item() == pytest.approx(0.4)


def test_distance_garde_la_forme_n_1() -> None:
    x, y = torch.rand(7, 1), torch.rand(7, 1)
    assert distance_to_center(x, y, DISQUE).shape == (7, 1)


def test_theta_initial_vaut_1_au_centre_et_0_loin() -> None:
    x_c, y_c = _point(0.5, 0.5)
    x_far, y_far = _point(0.05, 0.05)
    assert theta_initial(x_c, y_c, DISQUE).item() == pytest.approx(1.0, abs=1e-6)
    assert theta_initial(x_far, y_far, DISQUE).item() == pytest.approx(0.0, abs=1e-6)


def test_theta_initial_vaut_un_demi_sur_le_bord() -> None:
    # Sur le bord (d = R), tanh(0) = 0 donc θ₀ = ½, pour les deux formes.
    x, y = _point(0.6, 0.5)
    assert theta_initial(x, y, DISQUE).item() == pytest.approx(0.5, abs=1e-5)  # float32
    x, y = _point(0.6, 0.6)  # coin du pavé : Chebyshev = R, euclidienne = R√2
    assert theta_initial(x, y, PAVE).item() == pytest.approx(0.5, abs=1e-5)
    assert theta_initial(x, y, DISQUE).item() < 0.01


def test_theta_initial_est_borne_entre_0_et_1() -> None:
    x, y = torch.rand(1000, 1), torch.rand(1000, 1)
    theta = theta_initial(x, y, DISQUE)
    assert theta.shape == (1000, 1)
    assert torch.all(theta >= 0) and torch.all(theta <= 1)


def test_epsilon_regle_la_largeur_de_transition() -> None:
    # À d = R + ε on doit avoir θ₀ = ½(1 − tanh 1) ≈ 0.119, quel que soit ε.
    for eps in (0.005, 0.02):
        cfg = replace(DISQUE, EPS_IC=eps)
        x, y = _point(0.5 + 0.1 + eps, 0.5)
        assert theta_initial(x, y, cfg).item() == pytest.approx(0.5 * (1 - np.tanh(1)), abs=1e-5)


def test_grille_initiale_forme_et_symetrie() -> None:
    grid = initial_condition_grid(DISQUE)
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (CFG.FD_N, CFG.FD_N)
    assert grid.dtype == np.float64
    # Objet centré : symétrie par échange de x et y, et valeur maximale au centre
    assert np.allclose(grid, grid.T)
    assert grid[CFG.FD_N // 2, CFG.FD_N // 2] == pytest.approx(1.0, abs=1e-6)
    assert grid[0, :].max() < 1e-6  # θ₀ ≈ 0 sur les murs : compatible avec la BC


def test_conversions_theta_celsius() -> None:
    theta = torch.tensor([[0.0], [0.5], [1.0]])
    temp = theta_to_celsius(theta, CFG)
    assert torch.allclose(temp, torch.tensor([[20.0], [50.0], [80.0]]))
    assert torch.allclose(celsius_to_theta(temp, CFG), theta)


def test_grille_initiale_conserve_l_aire_de_l_objet() -> None:
    # Le lissage tanh est symétrique autour du bord : l'intégrale de θ₀_ε sur la pièce
    # vaut l'aire de l'objet, πR² pour le disque et (2R)² pour le pavé.
    dx = CFG.fd_dx
    aire_disque = initial_condition_grid(DISQUE).sum() * dx * dx
    aire_pave = initial_condition_grid(PAVE).sum() * dx * dx
    assert aire_disque == pytest.approx(np.pi * 0.1**2, rel=0.02)
    assert aire_pave == pytest.approx((2 * 0.1) ** 2, rel=0.02)
