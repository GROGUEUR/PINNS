"""Objet chaud (disque ou pavé) et condition initiale lissée θ₀_ε (Éq. 2 du sujet).

Sprint 1 — auteur A, relecteur B.

À implémenter (ROADMAP § 3, sprint 1) :
- distance d(x, y) au centre : euclidienne pour le disque, Chebyshev max(|x−cx|, |y−cy|)
  pour le pavé (R est alors le demi-côté) ;
- θ₀_ε(x, y) = ½ [1 − tanh((d − R) / ε)], vectorisé (NumPy et torch), forme (N, 1) ;
- conversions θ ↔ T en °C : T = T_amb + (T_obj − T_amb) · θ.
"""
