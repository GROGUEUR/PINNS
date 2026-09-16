# AGENTS.md — PINN : diffusion thermique 2D instationnaire

> Contexte pour les assistants IA (Claude, Copilot, Cursor, Codex…) travaillant sur ce dépôt.
> Lire ce fichier **en entier** avant toute modification. Les instructions explicites de l'utilisateur priment sur ce fichier.
> Plan de travail détaillé : `ROADMAP.md`. État d'avancement : § 10.

---

## 1. Règles d'or (non négociables)

1. **Chaque ligne doit pouvoir être expliquée à l'oral sans hésiter.** L'évaluation consiste à ce qu'un examinateur pointe une ligne et que l'étudiant l'explique. On préfère toujours **simple et lisible** à astucieux.
2. **Performant, sans être complexe.** Vectoriser (pas de boucle Python sur les points), mais pas de micro-optimisation obscure : ni `torch.compile`, ni `functorch`/`vmap`, ni JIT, ni CUDA custom.
3. **Chaque fonction a une docstring** (rôle, arguments avec **formes et unités**, retour, équation du sujet concernée). Chaque ligne non triviale a un commentaire qui dit **pourquoi**, pas quoi.
4. **Aucune constante magique.** Toute valeur physique ou tout hyperparamètre vit dans `src/config.py`.
5. **Pas de nouvelle dépendance** hors `torch`, `numpy`, `matplotlib`, `gradio`, `pytest`, sauf accord explicite du binôme.
6. **Ne pas réécrire un module entier** quand une modification locale suffit. Proposer un diff minimal.
7. **Ne jamais modifier** les tests pour les faire passer, ni les résultats dans `results/`, sans le signaler.
8. En cas de doute sur un choix physique ou méthodologique : **demander** plutôt que supposer.

---

## 2. Le sujet en bref

Prédire T(x, y, t) dans une pièce 2D de 1 m × 1 m avec un **PINN** (Raissi et al. 2019).

- **EDP (Éq. 1)** : ∂T/∂t − α (∂²T/∂x² + ∂²T/∂y²) = 0, avec α = 2·10⁻⁵ m²/s.
- **IC (Éq. 2)** : T = T_obj = 80 °C dans l'objet centré (disque ou pavé), T_amb = 20 °C ailleurs.
- **BC (Éq. 3)** : T = 20 °C sur les 4 murs, pour tout t (Dirichlet).
- **Perte (Éq. 4)** : L = w_IC·L_IC + w_BC·L_BC + w_res·L_res.
- L'objet **n'est pas une source** : il refroidit librement après t = 0.

**Travail imposé :**

1. Adimensionnement et points de collocation.
2. PINN PyTorch avec `torch.autograd`.
3. Entraînement Adam → L-BFGS, plus ≥ 1 technique avancée (RAR, poids dynamiques, hard constraints).
4. Solveur DF de référence, avec MSE et erreur L2 relative.
5. Démo Gradio : paramètres de l'objet, slider temporel, cartes d'erreur.

**Livrables :** dépôt Git documenté, présentation de 5 min, démonstrateur.

### Formulation adimensionnée (source de vérité, voir `docs/physics.md`)

```
x, y ∈ [0,1]          (x* = x/L, L = 1 m)
t*   = α t / L²       → temps caractéristique L²/α = 50 000 s
τ    = t* / T_STAR_MAX ∈ [0,1]        T_STAR_MAX = 0.1  (≈ 1,4 h réelle)
θ    = (T − T_amb)/(T_obj − T_amb) ∈ [0,1]

Résidu : r = ∂θ/∂τ − T_STAR_MAX · (θ_xx + θ_yy)
IC     : θ(x,y,0) = θ₀_ε(x,y) = ½ [1 − tanh((d(x,y) − R)/ε)]
         d = distance euclidienne (disque) | max(|x−cx|,|y−cy|) (pavé, R = demi-côté)
BC     : θ = 0 sur ∂Ω
Retour en °C : T = T_amb + (T_obj − T_amb) · θ
```

**Fait mesuré** (solveur DF, disque R = 0.1) : θ_max vaut 0.39 à t* = 0.005 et 0.017 à t* = 0.1. La dynamique est concentrée près de τ = 0, ce qui explique l'échantillonnage biaisé vers 0.

---

## 3. Stack et commandes

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch numpy matplotlib gradio pytest

pytest -q                                # tous les tests doivent passer avant un commit
python -m src.fd_solver                  # génère results/fd_reference.npz
python -m src.train --mode soft          # baseline (pertes pondérées)
python -m src.train --mode hard          # hard constraints
python -m scripts.evaluate --ckpt checkpoints/<nom>.pt
python app.py                            # démonstrateur Gradio
```

Python ≥ 3.10. Le code doit tourner **sur CPU** (GPU optionnel via `config.DEVICE`).

---

## 4. Arborescence et responsabilités

```
.
├── AGENTS.md  ROADMAP.md  README.md  requirements.txt
├── docs/physics.md        # dérivation de l'adimensionnement (LaTeX)
├── docs/walkthrough/      # un walkthrough par sprint et par auteur (voir § 5, « Fin de sprint »)
├── src/
│   ├── config.py          # constantes physiques + hyperparamètres (dataclass unique)
│   ├── geometry.py        # objet (disque/pavé), IC lissée θ₀_ε            [A]
│   ├── sampling.py        # points IC / BC / résidu, RAD                    [A, RAD: B]
│   ├── model.py           # MLP, variante hard constraints, paramétrique    [A]
│   ├── physics.py         # résidu EDP (autograd), L_IC, L_BC, L_res        [A]
│   ├── weighting.py       # pondération dynamique par normes de gradients   [B]
│   ├── train.py           # Adam → L-BFGS, logs, checkpoints                [A]
│   ├── fd_solver.py       # solveur de référence FTCS                       [B]
│   ├── metrics.py         # MSE, erreur L2 relative                         [B]
│   └── viz.py             # heatmaps, courbes                               [B]
├── scripts/evaluate.py    # checkpoint → métriques + figures                [B]
├── scripts/ablation.py    # tableau comparatif                              [B]
├── app.py                 # Gradio                                          [B]
├── tests/                 # pytest
├── checkpoints/  results/ # (checkpoints/ dans .gitignore sauf modèles de démo)
```

[A] = auteur principal « Modèle », [B] = auteur principal « Référence & Démo ». **Les deux doivent pouvoir expliquer tous les fichiers.**

---

## 5. Conventions de code (optimisées pour l'oral)

**Style**

- PEP 8, type hints partout, noms explicites (`theta_xx`, pas `u2`).
- Fonctions courtes (≤ ~40 lignes), une responsabilité chacune.
- Une idée par ligne : pas de compréhension imbriquée, pas d'opérateur ternaire chaîné.
- Les noms reprennent le sujet : `loss_ic`, `loss_bc`, `loss_res`, `w_ic`…

**Docstrings** (style NumPy, en français) :

```python
def pde_residual(model: nn.Module, xyt: torch.Tensor, t_star_max: float) -> torch.Tensor:
    """Résidu de l'équation de la chaleur adimensionnée (Éq. 1 du sujet).

    r = ∂θ/∂τ − t*_max · (∂²θ/∂x² + ∂²θ/∂y²)

    Args:
        model: réseau θ(x, y, τ).
        xyt: points de collocation, forme (N, 3), colonnes [x, y, τ], sans dimension.
        t_star_max: horizon temporel adimensionné (facteur issu du changement τ = t*/t*_max).

    Returns:
        Résidu, forme (N, 1). Vaut 0 si l'EDP est parfaitement satisfaite.
    """
    xyt = xyt.requires_grad_(True)            # on dérive θ par rapport aux ENTRÉES
    theta = model(xyt)                        # (N, 1)
    ones = torch.ones_like(theta)             # vecteur v du produit vecteur-jacobien

    # Dérivées premières ; create_graph=True pour pouvoir redériver (ordre 2)
    grad = torch.autograd.grad(theta, xyt, ones, create_graph=True)[0]   # (N, 3)
    theta_x, theta_y, theta_t = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]

    # Dérivées secondes : on ne garde que la composante utile de chaque gradient
    theta_xx = torch.autograd.grad(theta_x, xyt, ones, create_graph=True)[0][:, 0:1]
    theta_yy = torch.autograd.grad(theta_y, xyt, ones, create_graph=True)[0][:, 1:2]

    return theta_t - t_star_max * (theta_xx + theta_yy)
```

**Commentaires**

- Indiquer la **forme des tenseurs** en fin de ligne : `# (N, 1)`.
- Référencer l'équation du sujet (`# Éq. 4`) ou l'article (`# Wu et al. 2023, RAD`).
- Expliquer les choix non évidents : `# tanh : dérivée seconde non nulle, contrairement à ReLU`.

**Performance**

- Tous les points passent dans le réseau **en un seul batch**.
- Utiliser `torch.no_grad()` pour l'inférence et l'évaluation.
- Pas de `.item()` ni de `.cpu()` dans la boucle chaude, sauf pour le log tous les K pas.
- Solveur DF vectorisé par slicing NumPy (aucune boucle sur les nœuds).

**Fin de sprint : walkthrough obligatoire.** Chaque auteur rédige `docs/walkthrough/walkthrough_sprint_<n>_<A|B>.md` (exemple : `walkthrough_sprint_1_A.md`) avant la PR de fin de sprint. Contenu attendu : périmètre, ordre du travail, fonction par fonction avec les lignes clés et le **pourquoi**, constantes ajoutées, tests et régression que chacun attraperait, décisions à valider par l'autre, mesures faites, passage de relais, questions d'examinateur « doigt sur la ligne ». Le relecteur s'en sert pour l'explain-back.

**Commits** (Conventional Commits, en français) : `feat(physics): résidu EDP par autograd`, `fix(fd): pas de temps CFL`, `docs: …`, `test: …`.
Une PR = une fonctionnalité, relue par l'autre membre du binôme.

---

## 6. Décisions techniques verrouillées

| Sujet | Décision | Raison |
|---|---|---|
| Variables | Entrées (x, y, τ) ∈ [0,1]³, remises dans [−1,1] dans `forward` ; sortie θ | Échelles d'ordre 1, gradients stables |
| Horizon | `T_STAR_MAX = 0.1` | θ_max ≈ 0.017 : retour quasi complet à l'équilibre |
| Réseau de base | MLP 3 → [64]×4 → 1, **tanh**, init Xavier normal | C², simple, ~13k paramètres |
| IC | Lissée par tanh, `EPS_IC = 0.01` (à ajuster entre 0.005 et 0.02) | Un échelon n'est pas représentable par un MLP lisse |
| Échantillonnage | N_res = 20 000, N_IC = 5 000, N_BC = 4 000 ; τ = u² | Dynamique concentrée près de τ = 0 |
| Optimiseurs | Adam (lr 1e-3, décroissance exp.) ~20k it → L-BFGS (`lr=1`, `history_size=50`, `line_search_fn="strong_wolfe"`) | Exploration, puis convergence fine |
| L-BFGS | **Full-batch, points figés**, closure qui recalcule la perte | L-BFGS exige une perte déterministe |
| Hard constraints | θ̂ = θ₀_ε + τ · x(1−x) · y(1−y) · N(x,y,τ) | IC et BC exactes par construction |
| RAD | p ∝ \|r\|ᵏ / E[\|r\|ᵏ] + c, k = 1, c = 1, tous les 2 000 it | Défauts recommandés par Wu et al. 2023 |
| Poids dynamiques | λᵢ = Σ‖∇Lⱼ‖ / ‖∇Lᵢ‖, moyenne mobile 0.9, tous les 1 000 it | *Expert's Guide* (Wang et al. 2023) |
| Référence | FTCS explicite, grille 101×101, dt = 0.2·dx², **même θ₀_ε que le PINN** | Stable (dt ≤ dx²/4), comparaison équitable |
| Métriques | MSE et L2 relative **sur θ**, grille 101×101 × 51 instants ; erreur max en °C | Sur T, l'offset de 20 °C masquerait l'erreur |
| Démo | Champ complet pré-calculé à chaque changement de paramètre ; le slider t ne fait qu'indexer | Réponse instantanée |
| Seeds | Fixées dans `config.py` | Reproductibilité |

Toute modification de cette table doit être validée par le binôme puis reportée ici.

---

## 7. Pièges connus (à vérifier en priorité en cas de bug)

- Oubli de `requires_grad_(True)` sur les entrées, ou de `create_graph=True` : les dérivées valent `None` ou 0.
- ReLU ou GELU mal choisie : laplacien nul ou dégénéré. Rester en **tanh**.
- Facteur `T_STAR_MAX` oublié dans le résidu : la solution diffuse 10× trop vite.
- Rééchantillonnage pendant L-BFGS : divergence ou erreurs de line search.
- IC discontinue côté DF mais lissée côté PINN : fausse erreur près de t = 0.
- Erreur L2 relative calculée sur T : chiffres faussement bons.
- `detach()` manquant sur les poids adaptatifs ou causaux : ils seraient optimisés eux-mêmes.
- Formes (N,) et (N,1) mélangées : broadcasting silencieux vers (N,N). Toujours garder (N, 1).
- Instabilité du DF si dt > dx²/4.

---

## 8. Tests et validation

| Test | Critère |
|---|---|
| `test_fd.py` | Solution analytique sin(πx)·sin(πy)·exp(−2π²t*) : erreur L2 rel. < 1e-3 |
| `test_physics.py` | Résidu nul (à 1e-5 près) sur un « modèle » qui renvoie la solution analytique ci-dessus, avec le facteur T_STAR_MAX |
| `test_sampling.py` | Formes, bornes [0,1], points BC exactement sur les murs |
| `test_model.py` | Hard constraints : θ̂(·,·,0) = θ₀_ε et θ̂ = 0 sur les murs, à 1e-6 près |
| `test_metrics.py` | Erreur nulle pour des tableaux identiques ; valeur connue sur un cas simple |

**Objectifs indicatifs** (erreur L2 relative sur θ) : baseline soft < 20 %, avec L-BFGS et technique avancée < 5 %. On rapporte les chiffres obtenus, jamais des chiffres supposés.

---

## 9. Ce que l'agent ne doit PAS faire

- Introduire DeepXDE, PhysicsNeMo/Modulus, JAX, Lightning, Hydra ou W&B : le projet doit être « from scratch » en PyTorch.
- Générer de gros blocs de code non commentés, ou du code que le binôme ne pourrait pas expliquer.
- Changer une convention d'unités ou d'adimensionnement sans mettre à jour `docs/physics.md` et § 2.
- Inventer des résultats ou des métriques.
- Committer des checkpoints lourds, des `.npz` de plus de 10 Mo ou des notebooks non nettoyés.

**Quand l'agent écrit du code**, il doit :

1. dire dans quel fichier et pourquoi ;
2. fournir le diff minimal ;
3. ajouter ou mettre à jour le test ;
4. indiquer 2–3 questions qu'un examinateur pourrait poser sur ce code, avec leurs réponses.

---

## 10. État d'avancement (à mettre à jour à chaque fin de sprint, avec le walkthrough de § 5)

**Dernière mise à jour :** 2026-09-16 — par : Claude (assistant IA), à valider par le binôme

| Sprint | Contenu | Statut |
|---|---|---|
| 0 | Setup dépôt, config, lecture articles | ◐ code fait (arborescence, `requirements.txt`, `src/config.py` + tests, `docs/lectures.md`). Reste : lecture des articles par A et B, protection de `main` sur GitHub (voir README) |
| 1 | Adimensionnement, échantillonnage [A] · Solveur DF + tests [B] | ◐ A fait le 2026-09-16 (`docs/physics.md`, `geometry.py`, `sampling.py`, tests, `docs/walkthrough/walkthrough_sprint_1_A.md`) · B à faire (`fd_solver.py`, `test_fd.py`) |
| 2 | PINN baseline [A] · Métriques, viz, evaluate [B] → **M1** | ☐ |
| 3 | L-BFGS, hard constraints [A] · RAD, poids dynamiques, ablation [B] → **M2** | ☐ |
| 4 | PINN paramétrique [A] · Gradio [B] → **M3** | ☐ |
| 5 | Docs, slides, entraînement à l'oral | ☐ |
| 6+ | Bonus (voir ROADMAP § 5) | ☐ |

**Résultats actuels :** _(aucun, à remplir : config → erreur L2 rel. / MSE / temps)_

**Décisions en attente :**

- [ ] Forme par défaut de l'objet : disque (R = 0.1) ou pavé (côté 0.2) ? _Défaut provisoire dans `config.py` : disque, R = 0.1 (cas mesuré au § 2)._
- [ ] Paramètres variables dans la démo : (R, cx, cy) ou seulement R ?
- [ ] Densification de l'IC (choix de A, à valider par B) : 50 % des points IC tirés par rejet dans la bande |d − R| ≤ 3ε (`IC_EDGE_FRACTION`, `IC_EDGE_BAND_EPS`), τ uniforme pour la BC.

**Bloquants actuels :** _(aucun)_

---

## 11. Glossaire

- **PINN** : réseau dont la perte contient le résidu d'une EDP.
- **Point de collocation** : point (x, y, τ) où l'on évalue le résidu.
- **Résidu** : ce qui reste quand on injecte θ̂ dans l'EDP (0 = solution exacte).
- **Soft / hard constraints** : IC et BC imposées par la perte, ou par construction du réseau.
- **RAD / RAR-D** : rééchantillonnage des points là où le résidu est grand.
- **FTCS** : Forward-Time Centered-Space, schéma DF explicite.
- **CFL** : condition de stabilité du pas de temps (ici dt ≤ dx²/4).
- **Erreur L2 relative** : ‖θ̂ − θ_ref‖₂ / ‖θ_ref‖₂.

## 12. Références

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks*. J. Comput. Phys. 378.
- Wang, Sankaran, Wang, Perdikaris (2023). *An Expert's Guide to Training PINNs*. arXiv:2308.08468.
- Wang, Sankaran, Perdikaris. *Respecting causality for training PINNs*. arXiv:2203.07404.
- Wu, Zhu, Tan, Kartha, Lu (2023). *A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs*. CMAME. arXiv:2207.10289.
- McClenny, Braga-Neto (2023). *Self-adaptive PINNs*. J. Comput. Phys.
- PyTorch : `torch.autograd.grad`, `torch.optim.LBFGS`. Gradio : `gr.Blocks`, `gr.Slider`.
