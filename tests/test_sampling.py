"""Tests de src/sampling.py : formes, bornes, BC sur les murs, biais temporel, densification IC."""

from dataclasses import replace

import pytest
import torch

from src.config import CFG, set_seeds
from src.geometry import distance_to_center
from src.sampling import (
    CollocationPoints,
    sample_bc_points,
    sample_collocation_points,
    sample_ic_points,
    sample_residual_points,
    sample_tau,
)


def _dans_le_cube_unite(xyt: torch.Tensor) -> bool:
    return bool(torch.all(xyt >= 0) and torch.all(xyt <= 1))


# ---------------------------------------------------------------- IC

def test_ic_forme_et_tau_nul() -> None:
    ic = sample_ic_points(CFG)
    assert ic.shape == (CFG.N_IC, 3)
    assert torch.all(ic[:, 2:3] == 0)  # τ = 0 pour tous les points IC


def test_ic_dans_la_piece() -> None:
    assert _dans_le_cube_unite(sample_ic_points(CFG))


def test_ic_densifie_pres_du_bord_de_l_objet() -> None:
    ic = sample_ic_points(CFG)
    d = distance_to_center(ic[:, 0:1], ic[:, 1:2], CFG)
    dans_la_bande = (d - CFG.OBJECT_RADIUS).abs() <= CFG.ic_edge_band
    # Au moins la part demandée est dans la bande (les points uniformes en ajoutent un peu).
    assert dans_la_bande.float().mean().item() >= CFG.IC_EDGE_FRACTION


def test_ic_sans_densification_est_uniforme() -> None:
    cfg = replace(CFG, IC_EDGE_FRACTION=0.0)
    ic = sample_ic_points(cfg)
    d = distance_to_center(ic[:, 0:1], ic[:, 1:2], cfg)
    dans_la_bande = (d - cfg.OBJECT_RADIUS).abs() <= cfg.ic_edge_band
    assert dans_la_bande.float().mean().item() < 0.15  # aire de la bande ≈ 4 % de la pièce


# ---------------------------------------------------------------- BC

def test_bc_forme_et_bornes() -> None:
    bc = sample_bc_points(CFG)
    assert bc.shape == (CFG.N_BC, 3)
    assert _dans_le_cube_unite(bc)


def test_bc_points_exactement_sur_les_murs() -> None:
    bc = sample_bc_points(CFG)
    x, y = bc[:, 0], bc[:, 1]
    sur_un_mur = (x == 0) | (x == 1) | (y == 0) | (y == 1)
    assert torch.all(sur_un_mur)


def test_bc_repartis_a_parts_egales_sur_les_4_murs() -> None:
    bc = sample_bc_points(CFG)
    x, y = bc[:, 0], bc[:, 1]
    for mur in ((x == 0), (x == 1), (y == 0), (y == 1)):
        assert mur.sum().item() == CFG.N_BC // 4


def test_bc_tau_couvre_tout_l_horizon() -> None:
    tau = sample_bc_points(CFG)[:, 2]
    assert tau.min().item() < 0.05 and tau.max().item() > 0.95


# ---------------------------------------------------------------- Résidu

def test_res_forme_et_bornes() -> None:
    res = sample_residual_points(CFG)
    assert res.shape == (CFG.N_RES, 3)
    assert _dans_le_cube_unite(res)


def test_res_tau_concentre_pres_de_zero() -> None:
    tau = sample_residual_points(CFG)[:, 2]
    # Avec τ = u², P(τ < 0.05) = √0.05 ≈ 22 %, contre 5 % pour un tirage uniforme.
    assert (tau < 0.05).float().mean().item() > 0.15


def test_sample_tau_suit_u_puissance_exposant() -> None:
    n = 20_000
    assert sample_tau(n, replace(CFG, TAU_EXPONENT=1.0)).mean().item() == pytest.approx(0.5, abs=0.02)
    assert sample_tau(n, replace(CFG, TAU_EXPONENT=2.0)).mean().item() == pytest.approx(1 / 3, abs=0.02)
    assert sample_tau(n, CFG).shape == (n, 1)


def test_taille_personnalisee() -> None:
    assert sample_residual_points(CFG, n=123).shape == (123, 3)


# ---------------------------------------------------------------- Ensemble

def test_sample_collocation_points_regroupe_les_trois_familles() -> None:
    pts = sample_collocation_points(CFG)
    assert isinstance(pts, CollocationPoints)
    assert pts.ic.shape == (CFG.N_IC, 3)
    assert pts.bc.shape == (CFG.N_BC, 3)
    assert pts.res.shape == (CFG.N_RES, 3)


def test_tirage_reproductible_avec_set_seeds() -> None:
    set_seeds(7)
    a = sample_collocation_points(CFG)
    set_seeds(7)
    b = sample_collocation_points(CFG)
    assert torch.equal(a.ic, b.ic) and torch.equal(a.bc, b.bc) and torch.equal(a.res, b.res)
