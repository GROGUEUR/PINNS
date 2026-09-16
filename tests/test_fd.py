"""Tests pour le solveur par différences finies.

À vérifier (ROADMAP § 3, sprint 1) :
- Le solveur FTCS doit être précis par rapport à une solution analytique.
  Solution testée : θ(x,y,t*) = sin(πx)·sin(πy)·exp(−2π²t*)
  Erreur L2 relative < 1e-3.
"""

import numpy as np

from src.fd_solver import run_fd_solver

def test_fd_solver_analytical_solution():
    """Vérifie la précision du solveur par rapport à une solution analytique."""
    
    # Condition initiale analytique
    def analytical_ic(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)
        
    # Solution analytique à un instant t*
    def analytical_solution(X, Y, t_star):
        return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * np.pi**2 * t_star)
        
    # On exécute le solveur avec notre IC personnalisée
    t_save, X, Y, theta_save, _ = run_fd_solver(custom_theta_0=analytical_ic)
    
    # On calcule l'erreur L2 relative à l'instant final
    t_final = t_save[-1]
    theta_final_num = theta_save[-1]
    theta_final_ana = analytical_solution(X, Y, t_final)
    
    # Erreur L2 relative = ||θ_num - θ_ana||_2 / ||θ_ana||_2
    diff_norm = np.linalg.norm(theta_final_num - theta_final_ana)
    ref_norm = np.linalg.norm(theta_final_ana)
    
    rel_l2_error = diff_norm / ref_norm
    
    assert rel_l2_error < 1e-3, f"Erreur L2 relative trop élevée : {rel_l2_error:.2e}"
    print(f"Test réussi ! Erreur L2 relative : {rel_l2_error:.2e}")

if __name__ == "__main__":
    test_fd_solver_analytical_solution()

