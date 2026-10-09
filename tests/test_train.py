"""Tests de src/train.py : la boucle Adam fait baisser la perte, checkpoints rechargeables, CLI."""

from dataclasses import replace
from pathlib import Path

import torch

from src.config import CFG, Config, set_seeds
from src.model import build_model
from src.sampling import sample_collocation_points
from src.train import load_checkpoint, main, save_checkpoint, train_adam

# Petite configuration : quelques centaines de points pour que les tests restent rapides (~1 s).
PETIT = replace(CFG, N_RES=200, N_IC=100, N_BC=100, ADAM_ITERS=60, LOG_EVERY=10_000)


def _entrainer(cfg: Config) -> tuple[torch.nn.Module, torch.Tensor]:
    set_seeds(0)
    points = sample_collocation_points(cfg)
    model = build_model(cfg)
    history = train_adam(model, points, cfg)
    return model, history


def test_adam_fait_baisser_la_perte() -> None:
    _, history = _entrainer(PETIT)
    assert history.shape == (PETIT.ADAM_ITERS, 4)  # colonnes [L, L_IC, L_BC, L_res]
    # Mesuré avec cette graine : L passe de 0.225 à 0.135 en 40 pas (ratio 0.6). Sans optimizer.step(),
    # ou avec un gradient de mauvais signe, le ratio vaudrait 1 ou plus.
    assert history[-1, 0] < 0.75 * history[0, 0]


def test_historique_respecte_les_poids_de_la_config() -> None:
    # L = w_IC·L_IC + w_BC·L_BC + w_res·L_res (Éq. 4) avec les poids de la config, pas des 1 en dur.
    cfg = replace(PETIT, ADAM_ITERS=5, W_IC=2.0, W_BC=3.0, W_RES=5.0)
    _, history = _entrainer(cfg)
    total, ic, bc, res = history.unbind(dim=1)
    assert torch.allclose(total, 2 * ic + 3 * bc + 5 * res)


def test_checkpoint_aller_retour(tmp_path: Path) -> None:
    cfg = replace(PETIT, ADAM_ITERS=5, HIDDEN_LAYERS=(16, 16), OBJECT_SHAPE="pave")
    model, history = _entrainer(cfg)
    path = tmp_path / "essai.pt"
    save_checkpoint(path, model, cfg, history, train_time_s=1.5)

    loaded, cfg_loaded, ckpt = load_checkpoint(path)
    xyt = torch.rand(20, 3)
    with torch.no_grad():
        assert torch.equal(loaded(xyt), model(xyt))
    assert cfg_loaded == cfg  # même architecture, même objet : l'évaluation compare au bon DF
    assert torch.equal(ckpt["history"], history)
    assert ckpt["train_time_s"] == 1.5
    assert not loaded.training  # rechargé en mode évaluation


def test_main_ecrit_un_checkpoint_rechargeable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # CHECKPOINT_DIR est relatif : rien n'est écrit dans le dépôt
    path = main(["--adam-iters", "2", "--name", "essai"])
    assert path == Path("checkpoints") / "essai.pt"
    _, cfg_loaded, ckpt = load_checkpoint(path)
    assert cfg_loaded.ADAM_ITERS == 2  # l'option de la ligne de commande est bien appliquée
    assert ckpt["history"].shape == (2, 4)
