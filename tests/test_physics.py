"""Tests de src/physics.py : résidu de l'EDP par autograd (Éq. 1) et pertes (Éq. 4).

Les « modèles » ci-dessous sont des fonctions connues dont on a calculé les dérivées
à la main : ils remplacent le réseau pour vérifier les formules, pas l'apprentissage.
"""

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from src.config import CFG, set_seeds
from src.geometry import theta_initial
from src.physics import LossTerms, compute_losses, loss_bc, loss_ic, loss_res, pde_residual, total_loss
from src.sampling import sample_collocation_points


class SolutionAnalytique(nn.Module):
    """θ = sin(πx)·sin(πy)·exp(−2π²·t*_max·τ) : solution exacte (docs/physics.md § 9)."""

    def __init__(self, t_star_max: float) -> None:
        super().__init__()
        self.t_star_max = t_star_max

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x, y, tau = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        decay = torch.exp(-2 * math.pi**2 * self.t_star_max * tau)
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) * decay


class Polynome(nn.Module):
    """θ = x²·y + y³ + τ², donc θ_τ = 2τ, θ_xx = 2y, θ_yy = 6y (calculés à la main)."""

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x, y, tau = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        return x**2 * y + y**3 + tau**2


class Constante(nn.Module):
    """θ = c partout."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return torch.full_like(xyt[:, 0:1], self.value)


class ParaboleApprenable(nn.Module):
    """θ = a·(x² + y²) avec a appris : θ_τ = 0 et θ_xx + θ_yy = 4a."""

    def __init__(self, a: float) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x, y = xyt[:, 0:1], xyt[:, 1:2]
        return self.a * (x**2 + y**2)


class Theta0(nn.Module):
    """Renvoie exactement la condition initiale lissée θ₀_ε."""

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return theta_initial(xyt[:, 0:1], xyt[:, 1:2], CFG)


# ---------------------------------------------------------------- Résidu (Éq. 1)

def test_residu_nul_pour_la_solution_analytique() -> None:
    # Critère AGENTS § 8. Sans le facteur t*_max, on obtiendrait r = 1.8·π²·θ ≠ 0.
    set_seeds(0)
    xyt = torch.rand(500, 3, dtype=torch.float64)
    r = pde_residual(SolutionAnalytique(CFG.T_STAR_MAX), xyt, CFG.T_STAR_MAX)
    assert r.shape == (500, 1)
    assert r.abs().max().item() < 1e-5


@pytest.mark.parametrize(
    ("x", "y", "tau", "r_attendu"),
    [
        (0.3, 0.5, 0.25, 0.1),   # 2·0.25 − 0.1·8·0.5 = 0.5 − 0.4
        (0.9, 0.2, 0.0, -0.16),  # 0 − 0.1·8·0.2
        (0.0, 1.0, 1.0, 1.2),    # 2 − 0.8
    ],
)
def test_residu_d_un_polynome_calcule_a_la_main(x: float, y: float, tau: float, r_attendu: float) -> None:
    # r = θ_τ − t*_max·(θ_xx + θ_yy) = 2τ − 0.1·8y : vérifie chaque colonne (x, y, τ).
    xyt = torch.tensor([[x, y, tau]], dtype=torch.float64)
    assert pde_residual(Polynome(), xyt, 0.1).item() == pytest.approx(r_attendu, abs=1e-12)


def test_residu_ne_modifie_pas_les_points_de_l_appelant() -> None:
    xyt = torch.rand(10, 3)
    pde_residual(Polynome(), xyt, 0.1)
    assert not xyt.requires_grad


def test_residu_derivable_par_rapport_aux_poids() -> None:
    # Piège AGENTS § 7 : sans create_graph=True, L_res ne dépendrait plus des poids.
    # θ = a·(x² + y²) : r = −0.1·4a = −0.4a, L_res = 0.16·a² = 0.36 et dL_res/da = 0.32·a = 0.48 pour a = 1.5.
    model = ParaboleApprenable(1.5)
    loss = loss_res(model, torch.rand(50, 3), 0.1)
    loss.backward()
    assert loss.item() == pytest.approx(0.36, rel=1e-5)
    assert model.a.grad.item() == pytest.approx(0.48, rel=1e-5)


# ---------------------------------------------------------------- Pertes (Éq. 4)

def test_perte_ic_nulle_si_le_modele_vaut_theta0() -> None:
    xyt_ic = torch.rand(100, 3)
    xyt_ic[:, 2] = 0.0
    assert loss_ic(Theta0(), xyt_ic, CFG).item() == pytest.approx(0.0, abs=1e-12)


def test_perte_ic_mesure_l_ecart_a_theta0() -> None:
    # Centre de l'objet : θ₀ ≈ 1 ; coin de la pièce : θ₀ ≈ 0. Modèle nul : L_IC = (1² + 0²) / 2.
    xyt_ic = torch.tensor([[0.5, 0.5, 0.0], [0.05, 0.05, 0.0]])
    assert loss_ic(Constante(0.0), xyt_ic, CFG).item() == pytest.approx(0.5, abs=1e-6)


def test_perte_bc_est_l_ecart_quadratique_moyen_a_zero() -> None:
    # θ = 0.3 partout : L_BC = 0.3² = 0.09 (moyenne, pas somme).
    assert loss_bc(Constante(0.3), torch.rand(40, 3)).item() == pytest.approx(0.09, rel=1e-6)


def test_perte_residu_est_la_moyenne_des_carres() -> None:
    # Résidus du polynôme : 0.1 et −0.16, donc L_res = (0.01 + 0.0256) / 2 = 0.0178.
    xyt = torch.tensor([[0.3, 0.5, 0.25], [0.9, 0.2, 0.0]], dtype=torch.float64)
    assert loss_res(Polynome(), xyt, 0.1).item() == pytest.approx(0.0178, abs=1e-12)


def test_perte_totale_suit_l_equation_4() -> None:
    terms = LossTerms(ic=torch.tensor(1.0), bc=torch.tensor(2.0), res=torch.tensor(3.0))
    assert total_loss(terms, w_ic=10.0, w_bc=100.0, w_res=1000.0).item() == pytest.approx(3210.0)


def test_compute_losses_branche_chaque_famille_sur_sa_perte() -> None:
    # La solution analytique vérifie l'EDP et s'annule sur les murs, mais elle ne vaut pas θ₀ :
    # on attend L_res ≈ 0, L_BC ≈ 0 et L_IC nettement positive.
    cfg = replace(CFG, N_RES=300, N_IC=200, N_BC=200)
    set_seeds(0)
    points = sample_collocation_points(cfg)
    terms = compute_losses(SolutionAnalytique(cfg.T_STAR_MAX), points, cfg)
    assert terms.res.item() < 1e-8
    assert terms.bc.item() < 1e-10
    assert terms.ic.item() > 1e-2
