"""Boucle d'entraînement : Adam (décroissance exponentielle) puis L-BFGS, logs, checkpoints.

Sprint 2 (Adam) et 3 (L-BFGS) — auteur A, relecteur B.
Usage prévu : python -m src.train --mode soft | hard

À implémenter (AGENTS.md § 6) :
- Adam lr ADAM_LR pendant ADAM_ITERS itérations, tous les points en un seul batch ;
- L-BFGS full-batch sur des points figés, closure qui recalcule la perte ;
- journalisation tous les LOG_EVERY pas, sauvegarde dans CHECKPOINT_DIR.
"""
