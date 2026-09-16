"""Résidu de l'EDP par autograd et termes de la perte L_IC, L_BC, L_res (Éq. 1 et 4).

Sprint 2 — auteur A, relecteur B.

À implémenter (voir l'exemple commenté dans AGENTS.md § 5) :
- pde_residual : r = ∂θ/∂τ − T_STAR_MAX · (θ_xx + θ_yy), avec requires_grad_ sur les
  entrées et create_graph=True pour les dérivées secondes ;
- les trois pertes en MSE, chacune de forme scalaire, à partir de tenseurs (N, 1).
"""
