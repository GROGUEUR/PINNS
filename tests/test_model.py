"""Tests de src/model.py : forme de sortie, architecture, entrées dans [−1, 1], init Xavier, tanh."""

import math
from dataclasses import replace

import pytest
import torch

from src.config import CFG, set_seeds
from src.model import MLP, build_model, to_symmetric_range


def _nombre_de_parametres(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_sortie_de_forme_n_1() -> None:
    # Toujours (N, 1) : une sortie (N,) provoquerait un broadcasting (N, N) dans les pertes.
    assert MLP()(torch.rand(17, 3)).shape == (17, 1)


def test_architecture_personnalisee_compte_les_bons_parametres() -> None:
    # 3 → 5 → 7 → 1 : (3·5 + 5) + (5·7 + 7) + (7·1 + 1) = 20 + 42 + 8 = 70
    assert _nombre_de_parametres(MLP(hidden_layers=(5, 7))) == 70


def test_build_model_suit_la_config() -> None:
    cfg = replace(CFG, HIDDEN_LAYERS=(5, 7))
    model = build_model(cfg)
    assert _nombre_de_parametres(model) == 70
    assert next(model.parameters()).device.type == torch.device(cfg.DEVICE).type


def test_entrees_remises_dans_moins_un_un() -> None:
    xyt = torch.tensor([[0.0, 0.5, 1.0]])
    assert torch.equal(to_symmetric_range(xyt), torch.tensor([[-1.0, 0.0, 1.0]]))


def test_forward_recentre_les_entrees_puis_applique_tanh() -> None:
    # Réseau 3 → 3 → 1 réglé à la main : couche cachée = identité, sortie = 1re composante.
    # On attend θ̂ = tanh(2x − 1) : 0 pour x = 0.5, tanh(1) ≈ 0.7616 pour x = 1.
    model = MLP(hidden_layers=(3,))
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(3))
        model.layers[0].bias.zero_()
        model.layers[1].weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        model.layers[1].bias.zero_()
    out = model(torch.tensor([[0.5, 0.3, 0.3], [1.0, 0.3, 0.3]]))
    assert out[0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert out[1, 0].item() == pytest.approx(math.tanh(1.0), abs=1e-6)


def test_init_xavier_normal_et_biais_nuls() -> None:
    set_seeds(0)
    hidden = MLP().layers[1]  # couche 64 → 64 : Xavier normal donne σ = √(2 / (64 + 64)) = 0.125
    assert torch.all(hidden.bias == 0)
    # L'init par défaut de PyTorch donnerait σ ≈ 0.072 et des biais non nuls.
    assert hidden.weight.std().item() == pytest.approx(0.125, rel=0.1)


def test_tanh_donne_un_laplacien_non_nul() -> None:
    # Raison du choix de tanh (AGENTS § 6) : avec ReLU, ∂²θ/∂x² serait nul presque partout.
    set_seeds(0)
    model = MLP()
    xyt = torch.rand(64, 3, requires_grad=True)
    theta = model(xyt)
    theta_x = torch.autograd.grad(theta.sum(), xyt, create_graph=True)[0][:, 0]
    theta_xx = torch.autograd.grad(theta_x.sum(), xyt)[0][:, 0]
    assert theta_xx.abs().max().item() > 1e-3
