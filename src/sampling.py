"""Points de collocation : IC, BC et résidu (travail 1 du sujet), puis RAD.

Sprint 1 — auteur A, relecteur B. RAD au sprint 3 — auteur B.

À implémenter (ROADMAP § 3, sprint 1) :
- N_IC points (x, y, 0) avec densité renforcée près du bord de l'objet ;
- N_BC points sur les 4 murs, pour τ ~ U[0, 1] ;
- N_RES points intérieurs avec τ = u^TAU_EXPONENT pour concentrer près de τ = 0 ;
- toutes les sorties en torch.Tensor de forme (N, 3), colonnes [x, y, τ] ∈ [0, 1]³.
"""
