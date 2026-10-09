"""Métriques de validation : MSE et erreur L2 relative (travail 4 du sujet).

Sprint 2 — auteur B, relecteur A.

À implémenter (AGENTS.md § 6) :
- calculées sur θ, jamais sur T (l'offset de 20 °C masquerait l'erreur) ;
- erreur L2 relative = ‖θ̂ − θ_ref‖₂ / ‖θ_ref‖₂ sur la grille DF 101 × 101 × 51 ;
- erreur max en °C pour la lisibilité à l'oral.
"""

import numpy as np
from src.config import CFG

def mse(theta_pred: np.ndarray, theta_ref: np.ndarray) -> float:
    """Calcule l'erreur quadratique moyenne (MSE) sur θ.
    
    Args:
        theta_pred: Prédictions du modèle (adimensionnées).
        theta_ref: Référence (adimensionnée).
        
    Returns:
        Moyenne des carrés des écarts.
    """
    return float(np.mean((theta_pred - theta_ref)**2))

def relative_l2_error(theta_pred: np.ndarray, theta_ref: np.ndarray) -> float:
    """Calcule l'erreur L2 relative : ‖θ̂ − θ_ref‖₂ / ‖θ_ref‖₂.
    
    Calculée sur θ pour éviter que l'offset de température ambiante ne 
    masque l'erreur réelle.
    
    Args:
        theta_pred: Prédictions du modèle (adimensionnées).
        theta_ref: Référence (adimensionnée).
        
    Returns:
        Erreur L2 relative. Vaut 0 si les tableaux sont identiques.
    """
    diff_norm = np.linalg.norm(theta_pred - theta_ref)
    ref_norm = np.linalg.norm(theta_ref)
    
    if ref_norm == 0:
        if diff_norm == 0:
            return 0.0
        return float('inf')
        
    return float(diff_norm / ref_norm)

def max_error_celsius(theta_pred: np.ndarray, theta_ref: np.ndarray) -> float:
    """Calcule l'erreur absolue maximale en degrés Celsius.
    
    Args:
        theta_pred: Prédictions du modèle (adimensionnées).
        theta_ref: Référence (adimensionnée).
        
    Returns:
        Erreur maximale en °C.
    """
    max_theta_error = np.max(np.abs(theta_pred - theta_ref))
    # L'erreur en °C est simplement l'erreur sur θ multipliée par delta_T
    return float(max_theta_error * CFG.delta_t)
