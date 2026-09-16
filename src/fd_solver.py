"""Solveur de référence : différences finies explicites FTCS sur l'équation adimensionnée.

Sprint 1 — auteur B, relecteur A.
Usage prévu : python -m src.fd_solver  →  results/fd_reference.npz

À implémenter (ROADMAP § 3, sprint 1) :
- grille FD_N × FD_N, pas de temps fd_dt_star = FD_CFL_FACTOR · dx² (stable ssi ≤ dx²/4) ;
- laplacien par slicing NumPy, aucune boucle sur les nœuds ;
- même θ₀_ε lissée que le PINN (src/geometry.py), pour une comparaison équitable ;
- sauvegarde de FD_N_SAVE instants dans un .npz.
"""
