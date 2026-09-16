"""Réseau θ̂(x, y, τ) : MLP tanh, variante hard constraints, variante paramétrique.

Sprint 2 — auteur A, relecteur B. Hard constraints au sprint 3, paramétrique au sprint 4.

À implémenter (AGENTS.md § 6) :
- MLP 3 → HIDDEN_LAYERS → 1, activation tanh (C², contrairement à ReLU), init Xavier normal ;
- remise des entrées de [0, 1] vers [−1, 1] dans forward ;
- hard constraints : θ̂ = θ₀_ε(x, y) + τ · x(1−x) · y(1−y) · N(x, y, τ).
"""
