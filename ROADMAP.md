# ROADMAP — PINN : diffusion thermique 2D instationnaire

> Binôme : **A** = « Modèle » (PINN, entraînement) · **B** = « Référence & Démo » (solveur, métriques, Gradio)
> Durée supposée : **6 sprints d'environ 1 semaine**. Compressez ou étirez selon votre date de rendu.
> Livrables : dépôt Git propre + présentation de 5 min + démonstrateur Gradio.

---

## 0. Comprendre le sujet en 60 secondes

| Élément | Valeur |
|---|---|
| Domaine | Ω = [0,1] × [0,1] m, t ∈ [0, t_max] |
| EDP | ∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²), avec α = 2·10⁻⁵ m²/s (air) |
| Condition initiale | T = 80 °C dans l'objet (disque ou pavé centré), 20 °C ailleurs |
| Conditions aux limites | T = 20 °C sur les 4 murs (Dirichlet) |
| Sortie | T(x, y, t) partout, à tout instant |
| Imposé | PyTorch + autograd, Adam puis L-BFGS, ≥ 1 technique avancée, solveur DF/EF de référence, MSE + erreur L2 relative, démo Gradio |

L'objet n'est **pas** une source maintenue : il existe seulement dans la condition initiale, puis il refroidit.

### Adimensionnement retenu (à justifier à l'oral)

```
x* = x / L          y* = y / L          (L = 1 m)
t* = α t / L²       (temps de diffusion L²/α = 50 000 s ≈ 13,9 h)
θ  = (T − T_amb) / (T_obj − T_amb)   ∈ [0, 1]
τ  = t* / t*_max    ∈ [0, 1]         (entrée du réseau)

EDP  :  ∂θ/∂τ = t*_max · (∂²θ/∂x*² + ∂²θ/∂y*²)
IC   :  θ(x*, y*, 0) = 1 dans l'objet, 0 ailleurs
BC   :  θ = 0 sur ∂Ω
```

**Mesure faite avec un solveur DF de contrôle** (disque de rayon 0,1 m) :

| t* | θ_max | T_max | temps réel |
|---|---|---|---|
| 0.001 | 0.92 | 75.1 °C | 50 s |
| 0.005 | 0.39 | 43.6 °C | 4 min |
| 0.01 | 0.22 | 33.3 °C | 8 min |
| 0.05 | 0.05 | 22.8 °C | 42 min |
| **0.1** | **0.017** | **21.0 °C** | **1,4 h** |

**Conclusions :**

- On retient **t*_max = 0.1**. À cet instant, la pièce est presque revenue à l'équilibre.
- Toute la dynamique intéressante se joue sur **les 5 premiers % du temps**. C'est la « raideur » dont parle le sujet. Il faut donc échantillonner plus de points près de τ = 0.

---

## 1. Ce qu'on reprend de l'état de l'art

| Source | Ce qu'on en retient |
|---|---|
| Raissi et al. 2019 (article de référence du sujet) | Formulation de la perte IC + BC + résidu, entraînement Adam puis L-BFGS |
| Wang, Sankaran, Wang, Perdikaris 2023 — *An Expert's Guide to Training PINNs* | Adimensionnement ; **pondération par normes de gradients** λ̂ᵢ = Σ‖∇Lⱼ‖ / ‖∇Lᵢ‖, lissée par moyenne mobile (α = 0.9, tous les ~1000 itérations) ; entraînement causal (ε = 1) ; Fourier features |
| Wang, Sankaran, Perdikaris — *Respecting causality* | wᵢ = exp(−ε Σ_{k<i} L_r(t_k)) : un instant n'est appris qu'une fois le passé bien appris. C'est le bonus idéal pour un problème raide en temps. |
| Wu, Lu et al. 2023 — *Comprehensive study of adaptive sampling* (lu-group/pinn-sampling) | **RAD** : p(x) ∝ εᵏ / E[εᵏ] + c, défaut k = 1, c = 1 ; **RAR-D** : défaut k = 2, c = 0 |
| PINN-2DT (arXiv 2310.03755) | Code PyTorch pour problèmes 2D instationnaires. Les auteurs signalent la difficulté des discontinuités : cela confirme qu'il faut **lisser la condition initiale**. |
| AlirezaSamari/physics-informed-heat-equation, DiogoRibeiro7/pinn | Exemples de structure de dépôt et de visualisations |

---

## 2. Vue d'ensemble

```
Sprint 0 ─ Setup commun ─────────────────────────────── (A+B)
Sprint 1 ─ Fondations       A: adimensionnement + échantillonnage   B: solveur DF + tests
Sprint 2 ─ PINN baseline    A: modèle + résidu + Adam               B: métriques + visualisation
            ▶ Jalon M1 : première carte d'erreur PINN vs DF
Sprint 3 ─ Optimisation     A: L-BFGS + hard constraints            B: RAD + poids dynamiques
            ▶ Jalon M2 : tableau d'ablation
Sprint 4 ─ Démo             A: PINN paramétrique (r, cx, cy)        B: application Gradio
            ▶ Jalon M3 : démo fonctionnelle de bout en bout
Sprint 5 ─ Finalisation     A+B: documentation, slides, entraînement à l'oral
Sprint 6+─ Bonus            à la carte (§ 5)
```

**Règle d'or du binôme : on code chacun son module, mais on sait tous les deux expliquer toutes les lignes.**
Chaque pull request est relue par l'autre. Le relecteur doit pouvoir réexpliquer le code à voix haute avant de valider (« explain-back »).

---

## 3. Détail des sprints

### Sprint 0 — Setup (A + B, 1–2 jours)

- [ ] Créer le dépôt GitHub : branche `main` protégée, une branche par fonctionnalité, PR obligatoire.
- [ ] `requirements.txt` : `torch`, `numpy`, `matplotlib`, `gradio`, `pytest`. Rien de plus sans justification.
- [ ] Mettre en place l'arborescence (voir `AGENTS.md` § 4) et `src/config.py`, qui contient toutes les constantes.
- [ ] Ajouter `AGENTS.md` et `ROADMAP.md` au dépôt.
- [ ] Lire tous les deux Raissi 2019 (sections 2–3) et la section 2 de l'*Expert's Guide*.
- [ ] Fixer les seeds (`torch.manual_seed`, `np.random.seed`) dans `config.py`.

### Sprint 1 — Fondations

| A — Modèle | B — Référence |
|---|---|
| `docs/physics.md` : rédiger l'adimensionnement complet (§ 0) | `src/fd_solver.py` : schéma explicite FTCS vectorisé NumPy, avec dt = 0.2·dx² (condition de stabilité CFL 2D : dt ≤ dx²/4) |
| `src/geometry.py` : IC lissée θ₀(x,y) = ½[1 − tanh((d − r)/ε)], où d est la distance euclidienne (disque) ou la distance de Chebyshev max(\|Δx\|,\|Δy\|) (pavé) | Grille 101×101 ; sauvegarde de ~51 instants dans un `.npz` |
| `src/sampling.py` : points IC, BC et résidu ; τ = u² pour concentrer les points près de 0 ; points IC plus denses autour du bord de l'objet | `tests/test_fd.py` : comparer à la solution analytique sin(πx)·sin(πy)·e^{−2π²t*} (erreur < 1e-3) |
| `tests/test_sampling.py` : formes des tenseurs, bornes, BC bien sur les bords | **Le solveur DF utilise le même `θ₀` lissé que le PINN**, pour une comparaison équitable |

**Définition de terminé :** `pytest` passe au vert, le solveur DF tourne en moins de 1 s et une animation DF est générée.

### Sprint 2 — PINN baseline (pertes « soft »)

| A — Modèle | B — Évaluation |
|---|---|
| `src/model.py` : MLP 3 → [64]×4 → 1, tanh, init Xavier, entrées remises dans [−1, 1] | `src/metrics.py` : MSE, erreur L2 relative ‖θ̂−θ‖/‖θ‖ **calculée sur θ, pas sur T** (sur T, l'offset de 20 °C masquerait l'erreur) |
| `src/physics.py` : résidu par `torch.autograd.grad` avec `create_graph=True`, puis L_IC, L_BC, L_res (Éq. 4) | `src/viz.py` : cartes θ̂, θ_DF, \|erreur\| côte à côte ; courbes de pertes ; erreur en fonction du temps |
| `src/train.py` : boucle Adam (lr 1e-3, ~20k itérations, décroissance exponentielle), journalisation, checkpoints | `scripts/evaluate.py` : checkpoint → métriques + figures dans `results/` |
| `tests/test_physics.py` : le laplacien calculé par autograd vaut celui d'une fonction connue | Fichier `results/baseline.md` avec les chiffres |

**▶ Jalon M1 :** première carte d'erreur PINN vs DF. Une erreur de 10–30 % est **normale** à ce stade, car l'IC est raide.

### Sprint 3 — Optimisation et technique avancée

| A | B |
|---|---|
| Ajouter **L-BFGS** après Adam : `torch.optim.LBFGS(lr=1, history_size=50, line_search_fn="strong_wolfe")`, avec une closure. **Points figés, en full-batch** : pas de rééchantillonnage pendant L-BFGS. | **RAD** (k = 1, c = 1) : tous les N itérations, tirer un grand pool de candidats, calculer le résidu, rééchantillonner selon p ∝ \|r\|ᵏ/E[\|r\|ᵏ] + c |
| **Hard constraints** : θ̂ = θ₀(x,y) + τ·x(1−x)·y(1−y)·N(x,y,τ). IC et BC deviennent exactes et seul L_res reste à minimiser. | **Pondération dynamique** par normes de gradients (*Expert's Guide*) sur la version « soft » |
| Mesurer le temps d'entraînement | Script d'ablation : baseline / +L-BFGS / +RAD / +poids dynamiques / hard constraints |

**▶ Jalon M2 :** tableau d'ablation (erreur L2 rel., MSE, temps) plus une figure. C'est le cœur de la présentation.

### Sprint 4 — Démonstrateur

| A — PINN paramétrique | B — Gradio |
|---|---|
| Étendre l'entrée : (x, y, τ, r, cx, cy) → θ. L'IC lissée dépend des paramètres ; on tire les paramètres au hasard à chaque batch. | `app.py` (`gr.Blocks`) : choix forme disque/pavé, sliders r, cx, cy, **slider t** |
| Plages raisonnables : r ∈ [0.05, 0.2], centre ∈ [0.3, 0.7] | Quand un paramètre change : **pré-calculer tout le champ** (101² × 51 points en un seul batch, < 1 s) et lancer le DF. Le slider t ne fait ensuite qu'indexer un tableau, donc la réponse est instantanée. |
| **Plan B** si le paramétrique ne converge pas : 3–4 checkpoints pré-entraînés proposés dans un menu | Afficher 3 panneaux : PINN, DF, \|erreur\|, en °C, avec une échelle de couleur fixe ; plus les métriques |

**▶ Jalon M3 :** `python app.py` fonctionne sur une machine neuve après `pip install -r requirements.txt`.

### Sprint 5 — Finalisation

- [ ] **Audit des commentaires** (voir `AGENTS.md` § 5) : chaque fonction a sa docstring (rôle, formes, unités), chaque ligne non évidente a un commentaire « pourquoi ».
- [ ] `README.md` : installation, commandes, résultats, GIF de la démo.
- [ ] Slides de 5 min (plan en § 6).
- [ ] **Entraînement à l'oral : 3 sessions « doigt sur la ligne »** (§ 7). A interroge B sur le code de A, et inversement.
- [ ] Tag `v1.0`, puis vérifier une installation propre depuis un clone neuf.

---

## 4. Répartition : récapitulatif

| Module | Auteur | Relecteur (doit savoir l'expliquer) |
|---|---|---|
| `config.py`, structure | A + B | — |
| `geometry.py`, `sampling.py` | A | B |
| `model.py`, `physics.py`, `train.py` | A | B |
| hard constraints, L-BFGS, PINN paramétrique | A | B |
| `fd_solver.py`, `metrics.py`, `viz.py` | B | A |
| RAD, poids dynamiques, ablation | B | A |
| `app.py` | B | A |
| docs, slides, README | A + B | — |

Charge estimée : environ 50/50. A porte le cœur ML, B porte la validation, les techniques avancées et la démo.

---

## 5. Bonus (après M3, par ordre de rapport valeur/effort)

1. **Deuxième source de chaleur**, par superposition. L'équation est linéaire et les BC sont homogènes en θ, donc θ = θ₁ + θ₂ si les objets ne se chevauchent pas. Réutilise le PINN paramétrique sans réentraînement : c'est un excellent argument à l'oral.
2. **Animation GIF/MP4** et courbe T(t) en un point cliqué.
3. **Entraînement causal** (Wang et al.), à comparer à RAD.
4. **Fourier features** en entrée (σ ∈ [1, 10]), pour les fronts raides.
5. **Problème inverse** : retrouver α à partir de quelques « capteurs » bruités. C'est un point fort classique des PINN.
6. **Autres conditions aux murs** : Neumann (mur isolé) ou Robin (convection h(T − T_ext)).
7. **Source maintenue** (objet à 80 °C en permanence) : condition de Dirichlet interne.
8. **α hétérogène** (fenêtre, mur épais).
9. **Benchmark de temps** : inférence PINN vs DF sur une grille fine, et déploiement sur Hugging Face Spaces.

---

## 6. Plan de la présentation (5 min)

| Temps | Slide |
|---|---|
| 0:00 | Problème, EDP, IC/BC (1 schéma) |
| 0:45 | PINN : réseau, autograd, pertes, adimensionnement |
| 1:45 | Difficulté : raideur de l'IC (table θ_max(t)) et solutions retenues |
| 2:30 | Validation : carte d'erreur et tableau d'ablation |
| 3:30 | **Démo live** (sliders) |
| 4:30 | Limites (convection ignorée, air) et perspectives |

---

## 7. Banque de questions pour l'oral (à maîtriser tous les deux)

- **Pourquoi tanh et pas ReLU ?** La dérivée seconde de ReLU est nulle presque partout, donc le laplacien serait nul.
- **Pourquoi `create_graph=True` ?** Pour pouvoir redériver (ordre 2) et rétropropager à travers les dérivées.
- **Pourquoi `requires_grad_(True)` sur les entrées ?** On dérive θ par rapport à x, y et t, pas seulement par rapport aux poids.
- **Pourquoi adimensionner ?** Sans cela, α = 2e-5 et t ≈ 5000 s donnent des échelles de gradients écrasées ou explosives. Avec, les entrées et sorties sont d'ordre 1.
- **Pourquoi Adam puis L-BFGS ?** Adam est robuste loin de l'optimum ; L-BFGS (quasi-Newton) converge finement près de l'optimum, mais il exige une perte déterministe (full-batch).
- **Pourquoi lisser l'IC ?** Un échelon n'est pas dérivable et un MLP lisse ne peut pas le représenter. ε est un compromis entre fidélité et entraînabilité.
- **Pourquoi échantillonner plus près de τ = 0 ?** θ_max passe de 1 à 0.39 en 5 % du temps.
- **Que garantissent les hard constraints ?** IC et BC sont satisfaites exactement par construction : le facteur τ s'annule à t = 0 et le facteur x(1−x)y(1−y) s'annule sur les murs.
- **Pourquoi l'erreur L2 relative sur θ et pas sur T ?** ‖T‖ contient l'offset de 20 °C, ce qui rendrait l'erreur artificiellement petite.
- **Pourquoi dt = 0.2·dx² ?** Le schéma explicite 2D est stable si dt ≤ dx²/4.
- **Quel est l'intérêt d'un PINN si le DF est plus rapide ?** Solution continue et maillage-free, problèmes inverses, paramétrique (une inférence au lieu d'une simulation par configuration).
- **Quelles sont les limites physiques ?** Dans l'air réel, la convection domine. Ce modèle correspond plutôt à un solide.

---

## Sources

- [Raissi, Perdikaris, Karniadakis 2019 — PINNs (JCP)](https://doi.org/10.1016/j.jcp.2018.10.045)
- [Wang et al. 2023 — An Expert's Guide to Training PINNs](https://arxiv.org/abs/2308.08468)
- [Wang, Sankaran, Perdikaris — Respecting causality](https://arxiv.org/abs/2203.07404) · [code](https://github.com/PredictiveIntelligenceLab/CausalPINNs)
- [Wu et al. 2023 — Adaptive sampling for PINNs](https://arxiv.org/abs/2207.10289) · [code](https://github.com/lu-group/pinn-sampling)
- [McClenny & Braga-Neto — Self-adaptive PINNs](https://dl.acm.org/doi/10.1016/j.jcp.2022.111722)
- [PINN-2DT — 2D transient problems](https://arxiv.org/html/2310.03755v2)
- [AlirezaSamari/physics-informed-heat-equation](https://github.com/AlirezaSamari/physics-informed-heat-equation) · [DiogoRibeiro7/pinn](https://github.com/DiogoRibeiro7/pinn)
- [Gradio Quickstart](https://gradio.app/guides/quickstart)
