# Walkthrough — Sprint 2, binôme B (métriques, visualisation, script d'évaluation)

> Date : 2026-10-09. Auteur : B (avec l'assistant IA). Relecteur : A.
> But de ce document : que A puisse ré-expliquer chaque ligne de B à voix haute (« explain-back »),
> et que B retrouve en 5 minutes ce qu'il a fait et pourquoi, avant l'oral.

## 1. Périmètre (ROADMAP § 3, sprint 2, colonne B)

| Livrable | Fichier | État |
|---|---|---|
| Calcul de la MSE, erreur L2 relative, erreur max en °C | `src/metrics.py` | fait, tests intégrés |
| Courbes de pertes, heatmaps, évolution de l'erreur temporelle | `src/viz.py` | fait |
| Script complet pour évaluer le checkpoint depuis `load_checkpoint` | `scripts/evaluate.py` | fait |
| Document des résultats de base | `results/baseline.md` | fait |
| Tests de cohérence pour les métriques | `tests/test_metrics.py` | fait, tous passent |

Non touché (car colonne A) : `model.py`, `physics.py`, `train.py`.

Commandes :

```bash
python -m pytest tests/test_metrics.py -q
python -m scripts.evaluate --ckpt checkpoints/baseline_soft.pt
```

## 2. Ordre dans lequel le travail a été fait

1. Lecture des walkthroughs du sprint 1 et du `walkthrough_sprint_2_A.md` rédigé par A pour bien comprendre comment les checkpoints sont sauvegardés et chargés.
2. Implémentation des métriques dans `src/metrics.py` de façon isolée.
3. Création des tests de sanity checks dans `tests/test_metrics.py` : vérification que les calculs correspondent bien à l'erreur voulue, en tenant compte des retours en Celsius.
4. Développement des fonctions de tracé matplotlib dans `src/viz.py`. Prise en compte d'un détail de A : l'historique a un point par itération, donc pas besoin de multiplier l'axe X par `LOG_EVERY`.
5. Adaptation du pipeline d'évaluation `scripts/evaluate.py`. Le script utilise `load_checkpoint` du `train.py` de A, récupérant le `model`, la `config` de l'entraînement et l'historique complet.
6. Validation par un mini-entraînement rapide de 10 itérations (car PyTorch autograd sur le laplacien est lent sur CPU). L'objectif était de tester la solidité de `evaluate.py`.
7. Rédaction de ce walkthrough.

## 3. `src/metrics.py`, fonction par fonction

### `mse` et `relative_l2_error`

- Le calcul de l'erreur MSE et l'erreur L2 relative **s'effectuent sur $\theta$** et non sur la température réelle $T$ en Celsius. Si ces métriques étaient calculées sur $T$, l'offset constant (ex: 20 °C ambiant) viendrait écraser le ratio de l'erreur, donnant des chiffres faussement bons (piège identifié dans `AGENTS.md` § 7).
- Pas de boucle sur la grille : l'erreur est calculée globalement ou temporellement via les fonctions numpy/torch vectorisées (opérations sur les tenseurs directement).

### `max_error_celsius`

- Contrairement aux métriques relatives, l'erreur absolue maximale est rapportée en degrés Celsius pour garder un sens physique concret lors de l'évaluation du modèle.
- La formule `T = T_amb + (T_obj - T_amb) * theta` est utilisée.

## 4. `src/viz.py`, fonction par fonction

### `plot_loss_curves`

- La fonction récupère l'historique de forme `(N_iters, 4)` provenant de `ckpt["history"]` (colonnes : L totale, L_IC, L_BC, L_res).
- Contrairement à des implémentations standards qui n'enregistrent la loss que tous les $K$ itérations, le code de A l'enregistre à *chaque* pas. L'axe des abscisses est donc simplement un `range(N_iters)`.
- Échelle logarithmique (log10) requise en ordonnée, car la perte varie sur plusieurs ordres de grandeur.

### `plot_heatmaps`

- Permet d'afficher la vérité terrain (référence DF), la prédiction du modèle PINN, et l'erreur absolue pour un instant $t$ donné, côte-à-côte.
- Crucial pour identifier visuellement les défauts (ex: le "dôme" chaud au centre à $t=0$ pointé par A).

### `plot_error_vs_time`

- Affiche l'évolution de l'erreur L2 relative globale au fil du temps.

## 5. `scripts/evaluate.py`

Le script regroupe le tout.
1. Il charge les données du solveur Différences Finies (`fd_reference.npz`).
2. Il utilise le `load_checkpoint` du binôme A (qui donne la config et l'état du réseau).
3. Il crée le lot de points spatio-temporel correspondant à la grille DF. Détail important géré lors du passage de relais : `fd_reference.npz` stocke les temps en $t^*$, le réseau (et `evaluate.py`) doit normaliser avec $\tau = t^* / t^*_{max}$.
4. Il évalue le modèle avec `torch.no_grad()` pour gagner du temps et de la mémoire, car aucun backward() n'est nécessaire.
5. Il calcule les métriques, génère les figures et crée (ou met à jour) le rapport Markdown final `results/baseline.md`.

## 6. Décisions prises par B, à valider par A

| Décision | Pourquoi |
|---|---|
| Affichage de l'axe temporel des courbes de perte en valeurs brutes d'itérations | A enregistre chaque loss à chaque itération. Un échantillonnage visuel (slicing type `[::50]`) est possible si matplotlib ralentit, mais le log complet est utile pour repérer des petits sauts/creux. |
| Métriques sur $\theta$ mais "max error" en °C | Cela respecte strictement le `AGENTS.md` (erreur L2 et MSE sur l'adimensionné pour éviter de masquer l'erreur, et une vraie mesure interprétable). |
| Tester le pipeline avec 10 itérations (Adam) | PyTorch et autograd CPU sont très lents (environ 2 heures sur la machine de A pour 20k itérations). J'ai validé la mécanique de bout en bout avec très peu de calculs. |

## 7. Les tests, et ce que chacun attraperait

| Test | Régression détectée |
|---|---|
| `test_mse_identical_arrays` | Formule mathématique cassée ou ordre (y_true, y_pred) perturbé. |
| `test_relative_l2_error_on_theta` | Formule calculée sur la mauvaise échelle, erreur non relative, division par zéro non gérée (cas d'un array vide ou 0 pur). |
| `test_max_error_celsius` | Renvoi de l'erreur d'adimensionnement au lieu du format cible physique, conversion thermique erronée ($T_{amb}$ oublié). |

## 8. Passage de relais à A (fin du Sprint 2)

Le sprint 2 est techniquement bouclé. L'infrastructure de l'évaluation est en place.
Pour le **Sprint 3 (M2)** :
- A devra introduire L-BFGS (qui demandera toute son attention sur l'optimisation des gradients fixes) et implémenter l'architecture "Hard Constraints".
- Pendant ce temps, (B), je devrai attaquer le rééchantillonnage de résidus (RAD), les pondérations dynamiques et le script d'ablation pour générer le grand tableau comparatif.

## 9. Questions d'examinateur, doigt sur la ligne

- **Pourquoi l'erreur L2 relative n'est-elle pas calculée sur la vraie température $T$ ?** 
  Parce que la température physique $T$ possède un offset de 20°C (la température ambiante). Si on a une prédiction à 21°C au lieu de 20°C, l'erreur absolue est 1°C. Si on divise cette erreur par la valeur de référence de $T$ (~20°C), l'erreur relative apparaît ridiculement faible (1/20 = 5%), alors que l'écart sur le phénomène qui diffuse ($\theta$) est immense, vu que la vérité terrain de $\theta$ est 0. Sur $\theta$, on mesure l'écart sur la dynamique pure de la chaleur.
- **Pourquoi le calcul des points d'évaluation passe par `torch.no_grad()` ?** 
  Parce qu'on n'entraîne pas le réseau. Sans cela, PyTorch maintiendrait un graphe de calcul massif en mémoire pour chaque prédiction, engendrant potentiellement un dépassement de la RAM ou des temps d'inférence catastrophiques sur une grille complète (N > 500 000).
- **Comment `evaluate.py` convertit-il le temps du DF pour le PINN ?**
  Le DF stocke ses instants en temps adimensionnel $t^*$. Le PINN attend $\tau = t^* / t^*_{max}$. Donc `evaluate.py` divise `t_save` par le paramètre `cfg.T_STAR_MAX` récupéré depuis le checkpoint.
- **Pourquoi ne pas avoir re-généré la baseline 20k sur la machine B ?**
  L'entraînement est coûteux sur CPU. A l'a déjà exécuté et obtenu une loss. L'évaluation (inférence simple et calculs numpy) prend en revanche moins de quelques secondes et est testée avec une baseline réduite factice (Adam 10 iters) pour certifier son fonctionnement, évitant de brûler 2h de CPU.

