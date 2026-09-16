# Walkthrough — Sprint 1, binôme A (adimensionnement, géométrie, échantillonnage)

> Date : 2026-09-16. Auteur : A (avec l'assistant IA). Relecteur : B.
> But de ce document : que B puisse ré-expliquer chaque ligne de A à voix haute (« explain-back »),
> et que A retrouve en 5 minutes ce qu'il a fait et pourquoi, avant l'oral.

## 1. Périmètre (ROADMAP § 3, sprint 1, colonne A)

| Livrable | Fichier | État |
|---|---|---|
| Adimensionnement complet | `docs/physics.md` | fait |
| IC lissée θ₀_ε, disque et pavé | `src/geometry.py` | fait, 10 tests |
| Points IC / BC / résidu, τ = u², IC densifiée au bord | `src/sampling.py` | fait, 14 tests |
| Tests formes, bornes, BC sur les murs | `tests/test_sampling.py`, `tests/test_geometry.py` | fait |
| Constantes nouvelles | `src/config.py` | 2 champs, 1 propriété, 2 garde-fous |

Commandes de vérification :

```bash
python -m pytest -q          # 33 tests, ~4 s
python -c "from src.sampling import sample_collocation_points; p = sample_collocation_points(); print(p.ic.shape, p.bc.shape, p.res.shape)"
```

## 2. Ordre dans lequel le travail a été fait

1. Lecture de `AGENTS.md` § 2 et § 6 (décisions verrouillées) et de `ROADMAP.md` § 0.
2. **Tests d'abord** (`tests/test_geometry.py`), exécutés pour les voir échouer, puis `src/geometry.py`.
3. Ajout des constantes de densification dans `src/config.py`, puis **tests** de `sampling.py`, puis le code.
4. Rédaction de `docs/physics.md` à partir des formules déjà validées par les tests.
5. Suite complète, chronométrage de l'échantillonnage, mise à jour de `AGENTS.md` § 10.

Pourquoi tests d'abord : un test écrit après le code passe toujours, il ne prouve rien.
Écrit avant, il décrit le comportement voulu (par exemple « θ₀ vaut ½ exactement sur le bord »)
et on l'a vu échouer, donc on sait qu'il détecte une régression.

## 3. `docs/physics.md` : les trois formules à savoir réciter

| Étape | Formule | Où elle sert |
|---|---|---|
| Variables | x* = x/L, t* = αt/L², θ = (T − T_amb)/(T_obj − T_amb) | partout |
| Équation en t* | ∂θ/∂t* = θ_xx + θ_yy | solveur DF (B) |
| Résidu en τ = t*/t*_max | r = ∂θ/∂τ − t*_max (θ_xx + θ_yy) | `physics.py` (sprint 2) |

Le point à ne pas rater à l'oral : **le facteur t*_max = 0,1 dans le résidu**. Il vient de la règle
de dérivation en chaîne ∂/∂t* = (1/t*_max) ∂/∂τ. Sans lui, le PINN simule une diffusion dix fois
trop rapide.

Le § 9 de `physics.md` donne la solution analytique sin(πx) sin(πy) e^(−2π² t*) : B l'utilise pour
`test_fd.py`, A l'utilisera pour `test_physics.py`.

## 4. `src/geometry.py`, fonction par fonction

### `distance_to_center` (lignes 18–36)

```python
dx = x - cfg.OBJECT_CX
dy = y - cfg.OBJECT_CY
if cfg.OBJECT_SHAPE == "disque":
    return torch.sqrt(dx**2 + dy**2)
return torch.maximum(dx.abs(), dy.abs())
```

- Disque : distance euclidienne, le bord est le cercle d = R.
- Pavé : distance de Chebyshev max(|Δx|, |Δy|). Ses lignes de niveau sont des carrés, donc
  d = R décrit exactement le bord d'un carré de demi-côté R. **Une seule formule tanh pour
  les deux formes**, c'est tout l'intérêt.
- Pas de `else` : `Config.__post_init__` refuse toute forme autre que `"disque"` ou `"pave"`.

### `theta_initial` (lignes 39–54)

```python
d = distance_to_center(x, y, cfg)
return 0.5 * (1.0 - torch.tanh((d - cfg.OBJECT_RADIUS) / cfg.EPS_IC))
```

- tanh(0) = 0 donc θ₀ = ½ sur le bord ; à d = R ± ε, θ₀ = ½(1 ∓ tanh 1) ≈ 0,88 et 0,12.
  La transition a une largeur d'environ 2ε, soit 2 mailles du DF avec ε = 0,01.
- Élément par élément : la même fonction sert au PINN (colonne (N, 1)) et au DF (grille 101 × 101).

### `initial_condition_grid` (lignes 57–71)

```python
coords = torch.linspace(0.0, 1.0, cfg.FD_N, dtype=torch.float64)
x, y = torch.meshgrid(coords, coords, indexing="ij")
return theta_initial(x, y, cfg).numpy()
```

- Convention **à respecter dans `fd_solver.py` et `metrics.py`** : `grid[i, j] = θ₀(x_i, y_j)`,
  c'est le mode `indexing="ij"`. Avec `"xy"` les axes seraient transposés.
- `float64` : le DF fait 5 000 pas de temps, on ne veut pas accumuler l'arrondi float32.
- Le DF de B doit appeler cette fonction et non recoder l'échelon : sinon l'erreur mesurée
  près de t = 0 serait celle du lissage, pas celle du PINN (piège AGENTS.md § 7).

### `theta_to_celsius`, `celsius_to_theta` (lignes 74–81)

T = T_amb + ΔT·θ et l'inverse, avec ΔT = `cfg.delta_t` = 60 °C. Servent aux cartes en °C de la démo.

## 5. `src/config.py` : ce qui a été ajouté

| Ligne | Ajout | Rôle |
|---|---|---|
| 71 | `IC_EDGE_FRACTION = 0.5` | part des points IC tirés près du bord de l'objet |
| 72 | `IC_EDGE_BAND_EPS = 3.0` | demi-largeur de la bande, en multiples de ε (tanh 3 ≈ 0,995 : au-delà, θ₀ est plat) |
| 152 | propriété `ic_edge_band` | = 3ε, dérivée et non recopiée : si on change ε, la bande suit |
| 176 | garde-fou `N_BC % 4 == 0` | un quart des points par mur |
| 178 | garde-fou `0 ≤ IC_EDGE_FRACTION ≤ 1` | évite un nombre de points négatif |

## 6. `src/sampling.py`, fonction par fonction

Convention commune : chaque famille est un tenseur (N, 3) de colonnes [x, y, τ] dans [0, 1]³,
créé sur `cfg.DEVICE`. La reproductibilité vient de `set_seeds()`.

### `sample_tau` (lignes 30–44)

```python
u = torch.rand(n, 1, device=cfg.DEVICE)
return u**cfg.TAU_EXPONENT
```

τ = u². Densité 1/(2√τ), infinie en 0 : P(τ < 0,05) = √0,05 ≈ 22 % au lieu de 5 %.
C'est la réponse à la raideur : θ_max passe de 1 à 0,39 pendant les 5 premiers % du temps.

### `_sample_near_object_edge` (lignes 47–63)

```python
kept = torch.empty(0, 2, device=cfg.DEVICE)
while kept.shape[0] < n:
    xy = torch.rand(n, 2, device=cfg.DEVICE)
    d = distance_to_center(xy[:, 0:1], xy[:, 1:2], cfg)
    in_band = ((d - cfg.OBJECT_RADIUS).abs() <= cfg.ic_edge_band).squeeze(1)
    kept = torch.cat([kept, xy[in_band]], dim=0)
return kept[:n]
```

- **Tirage par rejet** : on tire n candidats uniformes, on garde ceux à moins de 3ε du bord,
  on recommence tant qu'on n'en a pas n.
- La boucle `while` porte sur des **lots**, pas sur des points : chaque passage est vectorisé.
  Pour R = 0,1 la bande couvre ≈ 4 % de la pièce, il faut ≈ 13 passages, 8 ms au total.
- Ne connaît que `distance_to_center` : marche pour le disque, le pavé, et toute forme future.
- `squeeze(1)` : le masque booléen doit être de forme (n,) pour indexer les lignes de `xy`.

### `sample_ic_points` (lignes 66–86)

Concatène `round(0.5 · N_IC)` points de la bande et le reste uniforme, puis une colonne τ = 0.
La moitié uniforme n'est pas décorative : elle ancre l'intérieur de l'objet (θ = 1) et le fond (θ = 0).

### `sample_bc_points` (lignes 89–118)

Un quart des points par mur, construit mur par mur avec `zeros`/`ones` pour la coordonnée fixée
et un tirage indépendant `s` pour la coordonnée libre. τ est **uniforme** : la BC θ = 0 doit tenir
à tout instant et n'est pas plus dure près de τ = 0, où θ vaut déjà ≈ 0 loin de l'objet.

### `sample_residual_points` (lignes 121–136) et `sample_collocation_points` (139–145)

x, y uniformes, τ par `sample_tau`. Le paramètre `n` optionnel servira au RAD de B :
`sample_residual_points(cfg, n=cfg.RAD_POOL_SIZE)` tire le pool de candidats.
`sample_collocation_points` renvoie un `NamedTuple` `points.ic / .bc / .res`.

## 7. Les tests, et ce que chacun attraperait

| Test | Régression détectée |
|---|---|
| `test_distance_disque_est_euclidienne` / `..._pave_est_chebyshev` (triangle 3-4-5) | inversion des deux métriques |
| `test_theta_initial_vaut_un_demi_sur_le_bord` | erreur de signe ou de facteur ½ dans tanh |
| `test_epsilon_regle_la_largeur_de_transition` | ε oublié ou mal placé |
| `test_grille_initiale_conserve_l_aire_de_l_objet` | facteur 2 sur R, mauvaise métrique, mauvais `dx` |
| `test_grille_initiale_forme_et_symetrie` | mauvais `indexing`, θ₀ non nul sur les murs |
| `test_bc_points_exactement_sur_les_murs` | BC « presque » sur le mur (x = 0,999) |
| `test_bc_repartis_a_parts_egales_sur_les_4_murs` | un mur oublié |
| `test_res_tau_concentre_pres_de_zero` | exposant oublié (τ uniforme) |
| `test_sample_tau_suit_u_puissance_exposant` | moyenne 1/3 pour u², 1/2 pour u |
| `test_ic_densifie_pres_du_bord_de_l_objet` | bande ou fraction ignorées |
| `test_tirage_reproductible_avec_set_seeds` | générateur non global, seed ignorée |

Seul ajustement de test pendant le sprint : tolérance `abs=1e-5` sur « θ₀ = ½ au bord », car en
float32 0,6 − 0,5 ≠ 0,1 exactement. Le code n'a pas été modifié pour faire passer le test.

## 8. Décisions prises par A, à valider par B (listées dans AGENTS.md § 10)

| Décision | Alternative écartée | Pourquoi |
|---|---|---|
| Densification IC par rejet dans une bande | tirage polaire exact pour le disque | le rejet est agnostique à la forme, une seule fonction pour disque et pavé |
| 50 % des points IC dans la bande, bande = 3ε | 100 % dans la bande | il faut aussi des points pour l'intérieur de l'objet et le fond |
| τ uniforme pour la BC | τ = u² partout | la BC n'est pas raide ; on garde le biais pour le résidu, là où il compte |
| Grille DF en float64, indexation « ij » | float32, « xy » | précision sur 5 000 pas ; convention explicite à partager avec B |

## 9. Mesures faites

| Mesure | Valeur |
|---|---|
| Tests | 33 passés |
| `sample_collocation_points` (29 000 points) | 8 ms sur CPU |
| Part des points IC à moins de 3ε du bord | 0,52 (attendu ≥ 0,50) |
| Médiane de τ pour le résidu | 0,249 (attendu 0,25 pour u²) |

## 10. Passage de relais à B

- `fd_solver.py` doit importer `initial_condition_grid` et garder la convention `grid[i, j] = θ(x_i, y_j)`.
- `test_fd.py` peut prendre la solution analytique de `physics.md` § 9.
- Une fois le DF prêt, reconfirmer la table θ_max(t*) de `physics.md` § 7.

## 11. Questions d'examinateur, doigt sur la ligne

- **Pourquoi lisser l'IC ?** Un échelon n'est pas dérivable et un MLP tanh est C^∞ : il ne peut pas
  le représenter, la perte L_IC resterait bloquée. ε règle le compromis fidélité / entraînabilité.
- **`torch.maximum(dx.abs(), dy.abs())`, c'est quoi ?** La distance de Chebyshev : ses lignes de
  niveau sont des carrés, donc d = R est le bord d'un carré de demi-côté R.
- **Pourquoi `indexing="ij"` ?** Pour que `grid[i, j]` corresponde à (x_i, y_j) ; avec « xy »
  NumPy transpose les axes et le DF comparerait θ(x, y) à θ(y, x).
- **Pourquoi τ = u² et pas τ uniforme ?** θ_max passe de 1 à 0,39 pour τ < 0,05 : c'est là que
  l'EDP est dure. u² y place 22 % des points au lieu de 5 %.
- **Une boucle `while`, alors que les boucles sont proscrites ?** Elle itère sur des lots de n
  candidats vectorisés, pas sur des points. Une dizaine de passages, 8 ms.
- **Pourquoi `squeeze(1)` ?** `distance_to_center` renvoie (n, 1) pour respecter la convention
  du dépôt ; un masque booléen de lignes doit être (n,).
- **Que vaut l'intégrale de θ₀_ε ?** L'aire de l'objet, πR², car le lissage tanh est symétrique
  autour du bord. C'est ce que teste `test_grille_initiale_conserve_l_aire_de_l_objet`.
- **Pourquoi `ic_edge_band` est une propriété et pas une constante 0,03 ?** Pour qu'elle suive ε
  automatiquement quand on fera l'ablation sur `EPS_IC`.
