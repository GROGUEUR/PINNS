"""Tests de src/config.py : cohérence des constantes, garde-fous et reproductibilité."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from src.config import CFG, set_seeds


def test_temps_caracteristique_vaut_50000_s() -> None:
    # L²/α = 1 / 2e-5 = 50 000 s, et t_max = 0.1 × 50 000 = 5 000 s (ROADMAP § 0)
    assert CFG.t_char_s == pytest.approx(50_000.0)
    assert CFG.t_max_s == pytest.approx(5_000.0)


def test_pas_de_temps_df_respecte_la_cfl() -> None:
    # Schéma FTCS 2D stable ssi dt* ≤ dx²/4 (AGENTS.md § 7)
    assert CFG.fd_dt_star <= CFG.fd_dx**2 / 4


def test_facteur_cfl_trop_grand_est_refuse() -> None:
    with pytest.raises(ValueError):
        replace(CFG, FD_CFL_FACTOR=0.3)


def test_objet_hors_de_la_piece_est_refuse() -> None:
    with pytest.raises(ValueError):
        replace(CFG, OBJECT_CX=0.95)  # 0.95 + R = 1.05 > 1


def test_forme_inconnue_est_refusee() -> None:
    with pytest.raises(ValueError):
        replace(CFG, OBJECT_SHAPE="triangle")


def test_config_est_figee() -> None:
    # Personne ne doit pouvoir modifier CFG en place ; on passe par replace().
    with pytest.raises(FrozenInstanceError):
        CFG.EPS_IC = 0.02  # type: ignore[misc]


def test_set_seeds_rend_les_tirages_reproductibles() -> None:
    set_seeds(123)
    a_torch, a_np = torch.rand(3), np.random.rand(3)
    set_seeds(123)
    b_torch, b_np = torch.rand(3), np.random.rand(3)
    assert torch.equal(a_torch, b_torch)
    assert np.array_equal(a_np, b_np)
