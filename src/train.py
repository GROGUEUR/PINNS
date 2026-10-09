"""Entraînement du PINN : Adam avec décroissance du pas, journal, checkpoints.

Sprint 2 — auteur A, relecteur B. L-BFGS et le mode « hard » arrivent au sprint 3.

Usage :
    python -m src.train --mode soft                     # baseline : ADAM_ITERS itérations
    python -m src.train --mode soft --adam-iters 500    # essai rapide

Choix de la baseline : les points de collocation sont tirés une seule fois puis gardés
fixes, comme dans Raissi et al. (2019). Le rééchantillonnage adaptatif (RAD) est l'objet
du sprint 3 et se comparera à cette référence.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from src.config import CFG, Config, config_from_dict, config_to_dict, set_seeds
from src.model import build_model
from src.physics import compute_losses, total_loss
from src.sampling import CollocationPoints, sample_collocation_points


def train_adam(model: nn.Module, points: CollocationPoints, cfg: Config = CFG) -> torch.Tensor:
    """Minimise la perte de l'Éq. 4 par Adam, chaque famille de points en un seul batch.

    Le pas décroît par paliers : lr ← lr × ADAM_LR_DECAY_RATE tous les ADAM_LR_DECAY_STEPS
    itérations (décroissance exponentielle de l'Expert's Guide, Wang et al. 2023).

    Args:
        model: réseau θ̂, entraîné en place.
        points: points IC, BC et résidu, fixes pendant tout l'entraînement.
        cfg: configuration (ADAM_*, poids W_*, LOG_EVERY, DEVICE).

    Returns:
        Historique des pertes, forme (ADAM_ITERS, 4), colonnes [L, L_IC, L_BC, L_res], sur CPU.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.ADAM_LR_DECAY_STEPS, gamma=cfg.ADAM_LR_DECAY_RATE
    )
    # Pré-alloué sur le device du modèle : y copier les pertes n'impose aucune synchronisation
    # CPU/GPU, contrairement à un .item() à chaque pas (AGENTS § 5).
    history = torch.zeros(cfg.ADAM_ITERS, 4, device=cfg.DEVICE)
    start = time.perf_counter()

    for it in range(cfg.ADAM_ITERS):
        optimizer.zero_grad()                                     # sinon les gradients s'additionnent entre les pas
        terms = compute_losses(model, points, cfg)                # L_IC, L_BC, L_res
        loss = total_loss(terms, cfg.W_IC, cfg.W_BC, cfg.W_RES)  # Éq. 4
        loss.backward()                                           # gradients de L par rapport aux poids
        optimizer.step()
        scheduler.step()                                          # compte les pas pour faire décroître le lr

        # detach : sans lui, history garderait en mémoire le graphe de calcul de chaque pas
        history[it] = torch.stack([loss, terms.ic, terms.bc, terms.res]).detach()  # (4,)
        if it % cfg.LOG_EVERY == 0 or it == cfg.ADAM_ITERS - 1:
            _log_progress(it, history[it], scheduler.get_last_lr()[0], time.perf_counter() - start)

    return history.cpu()


def _log_progress(it: int, losses: torch.Tensor, lr: float, elapsed_s: float) -> None:
    """Affiche une ligne de journal : seule lecture des pertes côté CPU, tous les LOG_EVERY pas.

    Args:
        it: numéro de l'itération.
        losses: pertes de l'itération, forme (4,), [L, L_IC, L_BC, L_res].
        lr: pas d'apprentissage courant.
        elapsed_s: temps écoulé depuis le début de l'entraînement, en secondes.
    """
    total, ic, bc, res = losses.tolist()
    print(
        f"it {it:6d} | L {total:.3e} | IC {ic:.3e} | BC {bc:.3e} | res {res:.3e} "
        f"| lr {lr:.2e} | {elapsed_s:7.1f} s",
        flush=True,  # le journal s'affiche en direct, même redirigé vers un fichier
    )


def save_checkpoint(path: Path, model: nn.Module, cfg: Config, history: torch.Tensor, train_time_s: float) -> None:
    """Sauvegarde tout ce qu'il faut pour réévaluer le réseau sans le réentraîner.

    Args:
        path: fichier .pt de destination ; son dossier est créé si besoin.
        model: réseau entraîné.
        cfg: configuration de l'entraînement (architecture, objet, ε, horizon…).
        history: pertes renvoyées par train_adam, forme (n_iter, 4).
        train_time_s: durée de l'entraînement, en secondes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),  # poids et biais de chaque couche
        "config": config_to_dict(cfg),      # pour reconstruire le même réseau et le même θ₀_ε
        "history": history,
        "train_time_s": train_time_s,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path) -> tuple[nn.Module, Config, dict]:
    """Recharge un checkpoint écrit par save_checkpoint, prêt pour l'évaluation (scripts/evaluate.py).

    Args:
        path: fichier .pt.

    Returns:
        (model, cfg, checkpoint) : réseau en mode évaluation sur le device courant, sa
        configuration, et le dictionnaire brut (clés history et train_time_s notamment).
    """
    # weights_only=True : ne recharge que des données (tenseurs, nombres, str), jamais du code
    checkpoint = torch.load(path, map_location=CFG.DEVICE, weights_only=True)
    # DEVICE dépend de la machine, pas de l'expérience : on garde celui de la machine courante
    cfg = replace(config_from_dict(checkpoint["config"]), DEVICE=CFG.DEVICE)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg, checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit la ligne de commande.

    Args:
        argv: liste d'arguments ; None (usage normal) lit ceux de la ligne de commande.

    Returns:
        Namespace avec les champs mode, adam_iters et name.
    """
    parser = argparse.ArgumentParser(description="Entraîne le PINN de diffusion thermique 2D.")
    parser.add_argument("--mode", choices=["soft"], default="soft",
                        help="soft : IC et BC imposées par la perte (Éq. 4). Le mode hard arrive au sprint 3.")
    parser.add_argument("--adam-iters", type=int, default=CFG.ADAM_ITERS,
                        help=f"nombre d'itérations Adam (défaut : {CFG.ADAM_ITERS})")
    parser.add_argument("--name", default=None,
                        help="nom du checkpoint, sans extension (défaut : baseline_<mode>)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Entraîne le PINN et écrit son checkpoint dans CHECKPOINT_DIR.

    Args:
        argv: arguments de la ligne de commande ; None en usage normal.

    Returns:
        Chemin du checkpoint écrit.
    """
    args = parse_args(argv)
    cfg = replace(CFG, ADAM_ITERS=args.adam_iters)  # replace relance les garde-fous de Config
    set_seeds(cfg.SEED)

    points = sample_collocation_points(cfg)  # tirés une fois, fixes ensuite (baseline Raissi 2019)
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mode {args.mode} | {n_params} paramètres | {cfg.ADAM_ITERS} itérations Adam | device {cfg.DEVICE}",
          flush=True)

    start = time.perf_counter()
    history = train_adam(model, points, cfg)
    train_time_s = time.perf_counter() - start

    name = args.name if args.name is not None else f"baseline_{args.mode}"
    path = cfg.CHECKPOINT_DIR / f"{name}.pt"
    save_checkpoint(path, model, cfg, history, train_time_s)
    print(f"Checkpoint écrit : {path} (entraînement : {train_time_s:.0f} s)", flush=True)
    return path


if __name__ == "__main__":
    main()
