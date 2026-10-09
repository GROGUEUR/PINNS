# Walkthrough — Sprint 2, binôme A (réseau, résidu par autograd, boucle Adam)

> Date : 2026-10-09. Auteur : A (avec l'assistant IA). Relecteur : B.
> But de ce document : que B puisse ré-expliquer chaque ligne de A à voix haute (« explain-back »),
> et que A retrouve en 5 minutes ce qu'il a fait et pourquoi, avant l'oral.

## 1. Périmètre (ROADMAP § 3, sprint 2, colonne A)

| Livrable | Fichier | État |
|---|---|---|
| MLP 3 → [64]×4 → 1, tanh, Xavier, entrées remises dans [−1, 1] | `src/model.py` | fait, 7 tests |
| Résidu par autograd, pertes L_IC, L_BC, L_res (Éq. 1 et 4) | `src/physics.py` | fait, 12 tests |
| Boucle Adam, journal, checkpoints, ligne de commande | `src/train.py` | fait, 4 tests |
| Config sauvegardée dans les checkpoints | `src/config.py` | 2 fonctions, 1 test |
| Entraînement baseline complet | `checkpoints/baseline_soft.pt` | fait, chiffres au § 9 |

Non touché, car colonne B : `metrics.py`, `viz.py`, `scripts/evaluate.py`, `results/baseline.md`.

Commandes :

```bash
python -m pytest -q
python -m src.train --mode soft --adam-iters 200    # essai rapide
OMP_NUM_THREADS=4 python -m src.train --mode soft   # baseline complète (20 000 itérations)
```

## 2. Ordre dans lequel le travail a été fait

1. Lecture des walkthroughs du sprint 1 (A et B) et de `AGENTS.md` § 5 à 7.
2. **`pytest` plantait sur la machine de A avant toute modification**, dans `test_fd.py` de B :
   « OMP: Error #15 ». Cause : le numpy de conda (compilé avec MKL) et le torch de pip
   embarquent chacun leur runtime OpenMP ; le second à s'initialiser arrête le processus.
   Correction sans toucher au code : un venv pip hors de OneDrive, `C:\Users\franc\venvs\pinns`,
   comme le prévoit le README. Le contournement `KMP_DUPLICATE_LIB_OK=TRUE` a été écarté :
   Intel le documente comme pouvant produire des résultats faux.
3. TDD module par module, chaque test vu en échec avant d'écrire le code :
   `test_model.py` puis `model.py`, `test_physics.py` puis `physics.py`,
   `test_train.py` et `test_config.py` puis `config.py` et `train.py`.
4. Mesure du coût d'une itération selon le nombre de threads, puis entraînement complet,
   pendant lequel ce document a été rédigé.

## 3. `src/model.py`, fonction par fonction

### `to_symmetric_range` (lignes 16–28)

```python
return 2.0 * xyt - 1.0
```

[0, 1] devient [−1, 1]. tanh est centrée en 0 et quasi linéaire autour de 0 : des entrées
centrées évitent de démarrer dans sa zone saturée, où les gradients s'écrasent. La remise à
l'échelle est **interne au réseau** : partout ailleurs (échantillonnage, θ₀, DF), les points
restent dans [0, 1].

### `MLP.__init__` (lignes 43–51)

```python
sizes = (3, *hidden_layers, 1)
for n_in, n_out in zip(sizes[:-1], sizes[1:]):
    layers.append(nn.Linear(n_in, n_out))
self.layers = nn.ModuleList(layers)
```

- `zip(sizes[:-1], sizes[1:])` donne les paires (entrée, sortie) de chaque couche :
  (3, 64), trois fois (64, 64), puis (64, 1).
- `nn.ModuleList` et pas une liste Python : sinon PyTorch n'enregistre pas les poids,
  `model.parameters()` est vide et l'optimiseur n'apprend rien.
- Nombre de paramètres : 3·64 + 64 + 3·(64·64 + 64) + 64 + 1 = **12 801**, affiché au lancement.

### `_init_xavier` (lignes 53–61)

`nn.init.xavier_normal_` tire les poids avec une variance 2 / (n_in + n_out), et les biais
valent 0. L'amplitude du signal reste du même ordre d'une couche à l'autre, ce qui garde
tanh hors saturation. Gain 1, comme Raissi et al. Le test vérifie σ = 0,125 sur une couche
64 → 64 ; l'init par défaut de PyTorch donnerait σ ≈ 0,072 et des biais non nuls.

### `forward` (lignes 63–75)

```python
h = to_symmetric_range(xyt)
for layer in self.layers[:-1]:
    h = self.activation(layer(h))
return self.layers[-1](h)
```

Couche cachée = linéaire puis tanh ; la dernière couche est linéaire seule, comme pour toute
régression. Pourquoi tanh : le résidu contient des dérivées secondes, et celle de ReLU est
nulle presque partout, donc le laplacien du réseau serait nul (`test_tanh_donne_un_laplacien_non_nul`).

### `build_model` (lignes 78–90)

Point d'entrée unique, utilisé par l'entraînement et par `load_checkpoint` : les deux
construisent forcément la même architecture. Au sprint 3, c'est là que le mode « hard »
choisira la variante à hard constraints.

## 4. `src/physics.py`, fonction par fonction

### `pde_residual` (lignes 29–58) : le cœur du PINN

```python
xyt = xyt.detach().requires_grad_(True)                                # l. 44
theta = model(xyt)                                                     # (N, 1)
ones = torch.ones_like(theta)
grad = torch.autograd.grad(theta, xyt, ones, create_graph=True)[0]    # (N, 3)
theta_x, theta_y, theta_tau = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]
theta_xx = torch.autograd.grad(theta_x, xyt, ones, create_graph=True)[0][:, 0:1]
theta_yy = torch.autograd.grad(theta_y, xyt, ones, create_graph=True)[0][:, 1:2]
return theta_tau - t_star_max * (theta_xx + theta_yy)
```

- **`requires_grad_(True)`** : on dérive θ par rapport aux entrées (x, y, τ), pas seulement aux poids.
- **`detach()`**, écart volontaire à l'exemple de `AGENTS.md` § 5 : `requires_grad_` agit en place.
  Sans `detach()`, le tenseur de l'appelant (`points.res`) garderait `requires_grad=True` après
  l'appel. `detach()` crée une nouvelle feuille du graphe qui partage la même mémoire, sans copie.
- **`ones`** : `autograd.grad` calcule un produit vecteur-jacobien vᵀJ. Chaque θᵢ ne dépend que de
  son propre point, donc avec v = 1 on obtient exactement le gradient de chaque θᵢ par rapport à
  son point, pour les N points en un seul appel.
- **`create_graph=True`** sert deux fois : pour pouvoir redériver (ordre 2), et pour que
  L_res reste dérivable par rapport aux poids lors du `backward()`.
- **`[:, 0:1]` et pas `[:, 0]`** : `0:1` garde la forme (N, 1). Mélanger (N,) et (N, 1)
  provoque un broadcasting silencieux vers (N, N) (piège `AGENTS.md` § 7).
- **Dérivées secondes** : le gradient de θ_x donne [θ_xx, θ_xy, θ_xτ] ; on garde la colonne x.
  On calcule donc aussi θ_xy et θ_xτ pour rien : c'est le prix de la simplicité, sans `functorch`.
- **Le facteur `t_star_max`** vient du changement de variable τ = t*/t*_max (`docs/physics.md` § 4).

### `loss_ic`, `loss_bc`, `loss_res` (lignes 61–103)

- Toutes des **moyennes** de carrés (MSE), pas des sommes : la perte ne dépend pas du nombre de
  points, donc les poids w gardent le même sens si l'on change N_IC, N_BC ou N_res.
- `loss_ic` compare θ̂(x, y, 0) à `theta_initial`, **le même θ₀_ε que le solveur DF de B**.
- `loss_bc` : la cible vaut 0 (θ = 0 ⟺ T = T_amb), donc l'écart est simplement θ̂².

### `compute_losses` et `total_loss` (lignes 106–137)

- `compute_losses` renvoie un `LossTerms(ic, bc, res)` : chaque terme reste séparé et attaché au
  graphe. Les poids dynamiques de B (sprint 3) en ont besoin, car ils utilisent ‖∇L_i‖ terme par terme.
- `total_loss` reçoit les poids **explicitement** : `cfg.W_*` pour la baseline, les λᵢ de
  `weighting.py` au sprint 3, sans rien changer dans `physics.py`.

## 5. `src/train.py`, fonction par fonction

### `train_adam` (lignes 30–66)

```python
optimizer.zero_grad()
terms = compute_losses(model, points, cfg)
loss = total_loss(terms, cfg.W_IC, cfg.W_BC, cfg.W_RES)
loss.backward()
optimizer.step()
scheduler.step()
history[it] = torch.stack([loss, terms.ic, terms.bc, terms.res]).detach()
```

- **`zero_grad()` d'abord** : PyTorch additionne les gradients d'un `backward()` à l'autre.
- **`StepLR(step_size=2000, gamma=0.9)`** : lr = 10⁻³ × 0,9^⌊it/2000⌋, la décroissance
  exponentielle par paliers de l'*Expert's Guide*, exactement celle décrite dans `config.py`.
- **Points fixes** : `points` est tiré une fois dans `main` et réutilisé à chaque pas (Raissi 2019).
- **`history` pré-alloué** (ADAM_ITERS, 4), colonnes [L, L_IC, L_BC, L_res] : copier les pertes
  dans un tenseur n'impose aucune synchronisation, contrairement à `.item()` à chaque pas.
- **`.detach()` sur l'historique** : sans lui, `history` resterait relié au graphe de chaque pas,
  et la mémoire grossirait à chaque itération.
- **`_log_progress`** (lignes 69–83) : seule lecture des pertes côté CPU, tous les `LOG_EVERY` = 500 pas.

### `save_checkpoint` et `load_checkpoint` (lignes 86–123)

Le checkpoint contient `model_state` (poids), `config` (dictionnaire de la config),
`history` et `train_time_s`. Au rechargement :

- `weights_only=True` : `torch.load` ne reconstruit que des données (tenseurs, nombres, chaînes),
  jamais du code. C'est pour cela que les `Path` de la config sont stockés en `str`.
- `DEVICE=CFG.DEVICE` : le device dépend de la machine, pas de l'expérience. Un réseau entraîné
  sur GPU se recharge sur un poste sans GPU.
- `model.eval()` : sans effet sur un MLP pur, mais c'est la convention avant toute évaluation.

### `parse_args` et `main` (lignes 126–176)

`--mode soft` (seul mode pour l'instant, « hard » au sprint 3), `--adam-iters`, `--name`.
`replace(CFG, ADAM_ITERS=...)` relance les garde-fous de `Config`. Ensuite : graines, tirage
unique des points, réseau, entraînement, sauvegarde dans `checkpoints/<nom>.pt`.
`main` accepte une liste d'arguments, ce qui permet de la tester sans terminal.

## 6. `src/config.py` : ce qui a été ajouté

| Ligne | Ajout | Rôle |
|---|---|---|
| 202 | `config_to_dict(cfg)` | config en types simples, les `Path` en `str`, pour le checkpoint |
| 220 | `config_from_dict(fields)` | reconstruit une `Config` ; les garde-fous de `__post_init__` s'appliquent |

## 7. Les tests, et ce que chacun attraperait

| Test | Régression détectée |
|---|---|
| `test_sortie_de_forme_n_1` | sortie (N,) au lieu de (N, 1) |
| `test_architecture_personnalisee_compte_les_bons_parametres` (3 → 5 → 7 → 1 : 70 paramètres) | couche oubliée ou mal câblée |
| `test_build_model_suit_la_config` | `HIDDEN_LAYERS` ignoré, mauvais device |
| `test_entrees_remises_dans_moins_un_un` | mauvaise formule de remise à l'échelle |
| `test_forward_recentre_les_entrees_puis_applique_tanh` (réseau réglé à la main) | remise à l'échelle oubliée, ReLU au lieu de tanh, activation ajoutée sur la sortie |
| `test_init_xavier_normal_et_biais_nuls` | init par défaut de PyTorch conservée |
| `test_tanh_donne_un_laplacien_non_nul` | activation à dérivée seconde nulle (ReLU) |
| `test_residu_nul_pour_la_solution_analytique` (critère `AGENTS.md` § 8) | facteur t*_max oublié, signe faux |
| `test_residu_d_un_polynome_calcule_a_la_main` (3 points, r = 2τ − 0,8y) | colonnes x, y, τ interverties, mauvaise composante de la dérivée seconde |
| `test_residu_ne_modifie_pas_les_points_de_l_appelant` | `detach()` retiré |
| `test_residu_derivable_par_rapport_aux_poids` (dL/da = 0,48 calculé à la main) | `create_graph=False` |
| `test_perte_ic_nulle_si_le_modele_vaut_theta0`, `test_perte_ic_mesure_l_ecart_a_theta0` | mauvaise cible pour l'IC |
| `test_perte_bc_est_l_ecart_quadratique_moyen_a_zero` | somme au lieu de moyenne, cible non nulle |
| `test_perte_residu_est_la_moyenne_des_carres` | carré oublié, somme au lieu de moyenne |
| `test_perte_totale_suit_l_equation_4` | poids intervertis |
| `test_compute_losses_branche_chaque_famille_sur_sa_perte` | familles de points interverties |
| `test_adam_fait_baisser_la_perte` | `optimizer.step()` oublié, gradient de mauvais signe |
| `test_historique_respecte_les_poids_de_la_config` | poids `W_*` ignorés, colonnes de l'historique dans le désordre |
| `test_checkpoint_aller_retour` | poids ou config non sauvés, `Path` non converti, rechargement en mode entraînement |
| `test_main_ecrit_un_checkpoint_rechargeable` | option `--adam-iters` ignorée, mauvais nom de fichier |
| `test_config_dict_aller_retour` | `Path` non reconverti, champ perdu |

**Seule modification d'un test après coup** : `test_adam_fait_baisser_la_perte` tournait 150
itérations sur 800 points (11 s) avec le seuil « perte divisée par 2 ». Il tourne maintenant 60
itérations sur 400 points, avec le seuil « perte × 0,75 ». Mesure à l'appui : avec cette graine,
la perte passe de 0,225 à 0,135 en 40 pas (ratio 0,6), et elle ne serait divisée par 2 qu'au pas 94.
Le test passait déjà avant ; la modification ne sert qu'à le rendre rapide.

## 8. Décisions prises par A, à valider par B (reportées dans `AGENTS.md` § 10)

| Décision | Alternative écartée | Pourquoi |
|---|---|---|
| Points de collocation fixes pendant Adam | nouveau tirage à chaque pas (*Expert's Guide*) | baseline = Raissi 2019 ; le rééchantillonnage est justement l'objet du RAD au sprint 3 |
| `detach()` dans `pde_residual` | `requires_grad_` direct, comme l'exemple de `AGENTS.md` § 5 | pas d'effet de bord sur les points de l'appelant |
| Décroissance du lr par paliers (`StepLR`, × 0,9 tous les 2 000 pas) | décroissance lisse (`ExponentialLR`) | c'est la définition déjà écrite dans `config.py` |
| Historique des pertes à chaque itération, stocké dans le checkpoint | un point tous les 500 pas | courbes de pertes complètes pour `viz.py`, sans synchronisation |
| `load_checkpoint(path)` fourni par A | B recode le chargement | une seule définition du format, testée des deux côtés |
| Venv pip hors de OneDrive | Python de base de conda | conflit OpenMP entre numpy (conda) et torch (pip) |

## 9. Mesures faites

| Mesure | Valeur |
|---|---|
| Tests | 58 passés en 6,9 s (venv) |
| Paramètres du réseau | 12 801 |
| Coût d'une itération Adam, 29 000 points, sur secteur | 1,41 s avec 1 thread, 0,32 s avec 4 threads, 0,41 s avec 12 threads (défaut) |
| Entraînement baseline, 20 000 itérations, 4 threads | 7 181 s, soit 2 h, avec un bridage passager pendant un passage sur batterie |
| Perte finale | L = 1,33·10⁻², dont L_IC = 1,18·10⁻², L_BC = 3,6·10⁻⁵, L_res = 1,4·10⁻³ |

Évolution de la perte (journal complet dans `ckpt["history"]`) :

| Itération | 0 | 500 | 2 000 | 5 000 | 10 000 | 20 000 |
|---|---|---|---|---|---|---|
| L | 0,404 | 0,102 | 0,043 | 0,019 | 0,015 | 0,013 |
| L_IC | 0,289 | 0,098 | 0,039 | 0,017 | 0,014 | 0,012 |

Vers l'itération 500, le réseau est dans la solution triviale θ ≈ 0 : seul L_IC reste élevé.
Il en sort ensuite lentement, et L_IC domine la perte jusqu'au bout.

### Contrôle de cohérence contre le DF de B

Script hors dépôt, sur la grille 101 × 101 × 51 du solveur DF. **Les chiffres officiels
viendront de `metrics.py` (B)** ; ceux-ci servent seulement à vérifier que l'entraînement
apprend la bonne physique.

| τ | t | θ_max DF | θ_max PINN | Erreur L2 relative à cet instant |
|---|---|---|---|---|
| 0 | 0 s | 1,000 | 2,037 | 0,568 |
| 0,02 | 100 s | 0,709 | 0,962 | 0,346 |
| 0,06 | 300 s | 0,341 | 0,391 | 0,161 |
| 0,10 | 500 s | 0,222 | 0,232 | 0,093 |
| 0,20 | 1 000 s | 0,118 | 0,115 | 0,091 |
| 0,50 | 2 500 s | 0,048 | 0,045 | 0,101 |
| 1,00 | 5 000 s | 0,017 | 0,018 | 0,166 |

Sur toute la grille : erreur L2 relative sur θ de **0,352**, MSE de 2,2·10⁻⁴, erreur maximale
de **62 °C**, au centre de l'objet à t = 0. Au-delà de t ≈ 500 s, le PINN suit bien le DF ;
les premiers instants sont faux.

### Diagnostic : le cœur de l'objet manque de points IC

Sur les points IC de l'entraînement, répartis par région :

| Région | Points | Part de L_IC | Erreur RMS |
|---|---|---|---|
| Cœur, d < R − 3ε | 43, soit 0,9 % | 26,7 % | 0,61 |
| Bande du bord, \|d − R\| ≤ 3ε | 2 596, soit 52 % | 62,7 % | 0,12 |
| Fond, d > R + 3ε | 2 361, soit 47 % | 10,5 % | 0,05 |

À t = 0, le réseau apprend bien le bord (0,46 pour une cible de 0,5 sur le cercle d = R),
mais forme un dôme qui monte à 2,04 au centre. **Cause : l'échantillonnage IC du sprint 1**
met la moitié des points dans la bande du bord et l'autre moitié uniformément dans la pièce,
dont le cœur de l'objet ne représente qu'environ 1 %.

Expérience de contrôle : on reprend le réseau entraîné pour 500 itérations Adam.

| Reprise de 500 itérations | L2 relative | Centre à t = 0 (cible 1) | Centre à 100 s (DF 0,709) | Erreur max |
|---|---|---|---|---|
| Sans reprise | 0,352 | 2,04 | 0,96 | 62 °C |
| Mêmes points (témoin) | 0,359 | 2,04 | 0,96 | 62 °C |
| Avec 1 000 points IC ajoutés dans le disque | 0,340 | 1,54 | 0,80 | 32 °C |

Des itérations en plus ne changent rien ; des points dans le cœur divisent l'erreur maximale
par deux en 500 pas. L'erreur globale baisse peu : le cœur n'est pas la seule source d'erreur,
la raideur des premiers instants reste à traiter au sprint 3.

**Correction proposée, à décider avec B** : dans `_sample_near_object_edge`, tirer les points
dans tout le disque élargi, d ≤ R + 3ε, au lieu de la seule bande. Le cœur recevrait alors
environ 700 points au lieu de 43. Il faudrait adapter `test_ic_densifie_pres_du_bord_de_l_objet`,
puis relancer l'entraînement de 2 h. Avec les hard constraints du sprint 3, le problème
disparaît de lui-même, puisque l'IC y est exacte par construction.

## 10. Passage de relais à B (`metrics.py`, `viz.py`, `scripts/evaluate.py`)

```python
from pathlib import Path
from src.train import load_checkpoint

model, cfg, ckpt = load_checkpoint(Path("checkpoints/baseline_soft.pt"))
# model : réseau en mode évaluation ; cfg : config de l'entraînement (objet, ε, T_STAR_MAX…)
# ckpt["history"] : (20 000, 4), colonnes [L, L_IC, L_BC, L_res] ; ckpt["train_time_s"] : secondes
```

- **Temps** : `fd_reference.npz` stocke `t_save` en t*, le réseau attend τ = t*/t*_max.
  Il faut donc passer `t_save / cfg.T_STAR_MAX` en troisième colonne.
- **Évaluation en un seul batch** : les 51 × 101 × 101 = 520 251 points tiennent dans un tenseur
  (N, 3) en float32, à évaluer sous `torch.no_grad()` ; environ 1 s sur CPU.
- **Métriques sur θ, pas sur T** (`AGENTS.md` § 6), avec la même convention `grid[i, j] = θ(x_i, y_j)`.
- **Courbes de pertes** : `ckpt["history"]` contient une ligne par itération, à tracer en échelle log.
- **Le checkpoint n'est pas versionné** (`checkpoints/` est ignoré par Git) : soit le binôme
  l'ajoute au dépôt, soit B le régénère avec la commande du § 1 (≈ 2 h sur CPU).
- **Si `pytest` plante chez B avec « OMP: Error #15 »**, c'est le même conflit conda/pip qu'au § 2.

## 11. Questions d'examinateur, doigt sur la ligne

- **Pourquoi `torch.ones_like(theta)` dans `autograd.grad` ?** `autograd.grad` calcule un produit
  vecteur-jacobien vᵀJ. Chaque θᵢ ne dépend que de son propre point, donc avec v = 1 on obtient
  le gradient de chaque θᵢ par rapport à son point, pour tous les points en un seul appel.
- **Pourquoi `create_graph=True` ?** Pour deux raisons : redériver θ_x et θ_y (dérivées
  secondes), et rétropropager L_res jusqu'aux poids. Sans lui, L_res ne dépendrait plus des poids.
- **Pourquoi `detach()` avant `requires_grad_(True)` ?** `requires_grad_` modifie le tenseur
  en place ; `detach()` crée une nouvelle feuille qui partage la mémoire, et les points de
  l'appelant restent intacts.
- **Pourquoi `grad[:, 0:1]` et pas `grad[:, 0]` ?** `0:1` garde la forme (N, 1). Une forme (N,)
  combinée à (N, 1) donne silencieusement une matrice (N, N).
- **Pourquoi des moyennes et pas des sommes dans les pertes ?** La perte ne dépend alors pas du
  nombre de points, et les poids w_IC, w_BC, w_res gardent le même sens si l'on change N.
- **Pourquoi `optimizer.zero_grad()` à chaque pas ?** PyTorch additionne les gradients d'un
  `backward()` à l'autre ; sans remise à zéro, chaque pas utiliserait la somme de tous les gradients passés.
- **Pourquoi `.detach()` en remplissant `history` ?** Sinon l'historique resterait relié au graphe
  de calcul de chaque itération, et la mémoire grossirait à chaque pas.
- **Pourquoi `weights_only=True` au chargement ?** Un fichier `.pt` est un pickle, qui peut exécuter
  du code. `weights_only=True` n'accepte que des données ; c'est pour cela que la config y est
  stockée en types simples.
- **Pourquoi `nn.ModuleList` et pas une liste Python ?** Une liste ordinaire n'enregistre pas les
  couches : `model.parameters()` serait vide et l'optimiseur n'aurait rien à mettre à jour.
- **Pourquoi la perte stagne-t-elle vers 0,1 au début ?** Le réseau trouve d'abord la solution
  triviale θ ≈ 0, qui satisfait déjà la BC et le résidu. Seul L_IC reste élevé, car la bosse
  chaude est petite et raide. C'est la raideur de l'IC qu'attaquent les techniques du sprint 3.
- **Pourquoi des points fixes pendant Adam ?** C'est la baseline de Raissi et al. ; le
  rééchantillonnage adaptatif (RAD) sera comparé à cette référence au sprint 3, et L-BFGS
  exige de toute façon des points figés.
- **Pourquoi 4 threads vont plus vite que 12 ?** Le i5-1240P a 4 cœurs performance et 8 cœurs
  efficaces. Le travail est découpé en parts égales, donc les cœurs lents retardent les rapides.
- **Pourquoi le PINN prédit-il 142 °C au centre à t = 0 ?** L'IC n'y est imposée que par
  43 points sur 5 000 : le réseau ajuste le bord, où sont la plupart des points, et forme un
  dôme au centre. L'expérience du § 9 le confirme : ajouter des points dans le cœur réduit
  l'erreur maximale de moitié, alors que des itérations seules ne changent rien.
