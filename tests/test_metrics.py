"""Tests pour les métriques de validation.

Vérifie (ROADMAP § 8) :
- Erreur nulle pour des tableaux identiques
- Valeur connue sur un cas simple
"""

import numpy as np
from src.metrics import mse, relative_l2_error, max_error_celsius
from src.config import CFG

def test_metrics_identical_arrays():
    """Vérifie que les métriques valent 0 si les tableaux sont identiques."""
    a = np.random.rand(10, 10)
    
    assert mse(a, a) == 0.0
    assert relative_l2_error(a, a) == 0.0
    assert max_error_celsius(a, a) == 0.0

def test_metrics_known_values():
    """Vérifie les calculs sur un cas simple connu."""
    ref = np.ones((2, 2))
    pred = np.zeros((2, 2))
    
    # MSE : moy((0 - 1)^2) = 1.0
    assert np.isclose(mse(pred, ref), 1.0)
    
    # L2 rel : ||-1||_2 / ||1||_2 = sqrt(4) / sqrt(4) = 1.0
    assert np.isclose(relative_l2_error(pred, ref), 1.0)
    
    # Max error °C : max(abs(-1)) * delta_T = 1.0 * CFG.delta_t = 60.0 (si T_obj=80, T_amb=20)
    assert np.isclose(max_error_celsius(pred, ref), CFG.delta_t)

