# PINN — diffusion thermique 2D instationnaire dans une pièce

Prédiction de T(x, y, t) dans une pièce de 1 m × 1 m avec un *Physics-Informed Neural Network*
(Raissi et al. 2019), validé contre un solveur en différences finies et présenté dans une démo Gradio.

- Sujet : [PINNs_M2_Temperature_prediction.pdf](PINNs_M2_Temperature_prediction.pdf)
- Contexte, conventions et décisions techniques : [AGENTS.md](AGENTS.md)
- Plan de travail par sprint : [ROADMAP.md](ROADMAP.md)

## Installation

Python ≥ 3.10. Le code tourne sur CPU ; le GPU est utilisé automatiquement s'il est disponible.

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Astuce Windows/OneDrive : créer le `.venv` **hors** du dossier synchronisé (ou l'exclure de
la synchronisation), sinon OneDrive tente de synchroniser des milliers de fichiers.

## Commandes

```bash
pytest -q                                  # tests (doivent passer avant tout commit)
python -m src.fd_solver                    # référence DF  → results/fd_reference.npz   (sprint 1)
python -m src.train --mode soft            # PINN baseline                                (sprint 2)
python -m src.train --mode hard            # hard constraints                             (sprint 3)
python -m scripts.evaluate --ckpt checkpoints/<nom>.pt                                 # (sprint 2)
python -m scripts.ablation                 # tableau comparatif                           (sprint 3)
python app.py                              # démonstrateur Gradio                         (sprint 4)
```

## Arborescence

```
src/config.py     toutes les constantes (dataclass figée CFG) et set_seeds()
src/geometry.py   objet disque/pavé, IC lissée θ₀_ε                   [A]
src/sampling.py   points IC / BC / résidu, RAD                        [A, RAD : B]
src/model.py      MLP tanh, hard constraints, paramétrique             [A]
src/physics.py    résidu EDP par autograd, L_IC, L_BC, L_res           [A]
src/weighting.py  poids dynamiques par normes de gradients             [B]
src/train.py      Adam → L-BFGS, logs, checkpoints                     [A]
src/fd_solver.py  solveur de référence FTCS                            [B]
src/metrics.py    MSE, erreur L2 relative                              [B]
src/viz.py        heatmaps, courbes                                    [B]
scripts/          evaluate.py, ablation.py                             [B]
app.py            Gradio                                               [B]
tests/            pytest
docs/             physics.md (adimensionnement), lectures.md (fiches de lecture)
checkpoints/      modèles (ignorés par Git sauf demo_*.pt)
results/          métriques et figures (.npz ignorés par Git)
```

## Workflow Git

- `main` est protégée : une branche par fonctionnalité, une PR relue par l'autre membre du binôme.
- Commits en français, Conventional Commits : `feat(physics): résidu EDP par autograd`.
- Pour activer la protection sur GitHub (une seule fois, par le propriétaire du dépôt) :
  *Settings → Branches → Add branch protection rule*, pattern `main`, cocher
  *Require a pull request before merging* et *Require approvals (1)*.

## Résultats

_À compléter à partir du sprint 2 (jalon M1)._
