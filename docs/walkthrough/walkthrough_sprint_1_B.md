# Walkthrough: Sprint 1 (Rôle B)

Le rôle B du Sprint 1 (Fondations) a été complété avec succès. 

## Changements apportés

### `src/geometry.py`
- Implémentation des fonctions NumPy et PyTorch pour la condition initiale lissée (`theta_0_np` et `theta_0_pt`), nécessaires au solveur par différences finies et au modèle de base.
- Ajout de la fonction utilitaire `theta_to_T` pour la conversion adimensionné vers °C.

### `src/fd_solver.py`
- Implémentation d'un solveur explicite par différences finies (FTCS) entièrement vectorisé en NumPy.
- Le solveur utilise une grille spatiale de 101x101 nœuds. Le calcul du Laplacien se fait par slicing NumPy, évitant ainsi toute boucle chronophage sur les nœuds.
- La limite de stabilité temporelle (CFL) pour ce schéma 2D `dt* = 0.2 * dx^2` (où max admissible est `dx^2/4`) a été respectée.
- La sauvegarde est effectuée sur un nombre fixe d'instants (`FD_N_SAVE` = 51) tel que demandé. 

### `tests/test_fd.py`
- Création d'un test automatisé vérifiant la validité de l'implémentation FD. 
- La solution numérique est testée vis-à-vis d'une solution analytique précise `θ(x,y,t*) = sin(πx)·sin(πy)·exp(−2π²t*)`.

## Validation

- **Tests unitaires** : L'erreur relative $L_2$ du solveur mesurée face à la solution analytique est de `2.27e-04` (< `1e-3`), confirmant ainsi son excellente précision. 
- **Exécution** : Le script `python -m src.fd_solver` a été lancé et a généré avec succès l'archive `results/fd_reference.npz` en moins de 1 seconde (0.697 s), respectant la contrainte de temps d'exécution stipulée dans la roadmap.
