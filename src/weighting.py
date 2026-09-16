"""Pondération dynamique des pertes par normes de gradients (Wang et al. 2023).

Sprint 3 — auteur B, relecteur A.

À implémenter (AGENTS.md § 6) :
- λ̂ᵢ = Σⱼ‖∇_θ Lⱼ‖ / ‖∇_θ Lᵢ‖ calculé tous les WEIGHT_UPDATE_EVERY pas ;
- lissage λ ← WEIGHT_MOMENTUM · λ + (1 − WEIGHT_MOMENTUM) · λ̂ ;
- les poids sont détachés du graphe (piège AGENTS.md § 7 : sinon ils seraient optimisés).
"""
