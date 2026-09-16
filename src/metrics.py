"""Métriques de validation : MSE et erreur L2 relative (travail 4 du sujet).

Sprint 2 — auteur B, relecteur A.

À implémenter (AGENTS.md § 6) :
- calculées sur θ, jamais sur T (l'offset de 20 °C masquerait l'erreur) ;
- erreur L2 relative = ‖θ̂ − θ_ref‖₂ / ‖θ_ref‖₂ sur la grille DF 101 × 101 × 51 ;
- erreur max en °C pour la lisibilité à l'oral.
"""
