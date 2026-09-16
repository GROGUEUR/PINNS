"""Solveur de référence : différences finies explicites FTCS sur l'équation adimensionnée.

Sprint 1 — auteur B, relecteur A.
Usage prévu : python -m src.fd_solver  →  results/fd_reference.npz

À implémenter (ROADMAP § 3, sprint 1) :
- grille FD_N × FD_N, pas de temps fd_dt_star = FD_CFL_FACTOR · dx² (stable ssi ≤ dx²/4) ;
- laplacien par slicing NumPy, aucune boucle sur les nœuds ;
- même θ₀_ε lissée que le PINN (src/geometry.py), pour une comparaison équitable ;
- sauvegarde de FD_N_SAVE instants dans un .npz.
"""

import time
import numpy as np

from src.config import CFG, set_seeds
from src.geometry import initial_condition_grid, theta_to_celsius

def run_fd_solver(custom_theta_0=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exécute le solveur FTCS et renvoie les résultats.
    
    Args:
        custom_theta_0: Fonction optionnelle de condition initiale (utilisée pour les tests analytiques).
                        Si None, utilise theta_0_np de src.geometry.
                        
    Returns:
        tuple (t_save, X, Y, theta_save, T_save):
            - t_save: instants sauvegardés (FD_N_SAVE,)
            - X, Y: grilles 2D (FD_N, FD_N)
            - theta_save: historique θ (FD_N_SAVE, FD_N, FD_N)
            - T_save: historique T en °C (FD_N_SAVE, FD_N, FD_N)
    """
    # 1. Grille spatiale
    x = np.linspace(0, 1, CFG.FD_N)
    y = np.linspace(0, 1, CFG.FD_N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # forme (FD_N, FD_N)
    
    dx = CFG.fd_dx
    dy = dx  # grille carrée
    dx2 = dx**2
    dy2 = dy**2
    
    # 2. Paramètres temporels
    dt_star = CFG.fd_dt_star
    total_steps = int(np.ceil(CFG.T_STAR_MAX / dt_star))
    # Réajustement léger de dt_star pour tomber exactement sur T_STAR_MAX
    dt_star = CFG.T_STAR_MAX / total_steps
    
    # Instants de sauvegarde
    save_steps = np.linspace(0, total_steps, CFG.FD_N_SAVE, dtype=int)
    save_idx = 0
    
    # 3. Allocation et condition initiale
    if custom_theta_0 is not None:
        theta = custom_theta_0(X, Y)
    else:
        theta = initial_condition_grid()
        
    # Appliquer la condition aux limites (Dirichlet = 0 sur les bords)
    theta[0, :] = 0
    theta[-1, :] = 0
    theta[:, 0] = 0
    theta[:, -1] = 0
    
    theta_save = np.zeros((CFG.FD_N_SAVE, CFG.FD_N, CFG.FD_N))
    t_save = np.zeros(CFG.FD_N_SAVE)
    
    if 0 in save_steps:
        theta_save[save_idx] = theta.copy()
        t_save[save_idx] = 0.0
        save_idx += 1
        
    # 4. Boucle temporelle FTCS
    for n in range(1, total_steps + 1):
        # Laplacien par slicing : aucune boucle sur les noeuds
        # Laplacian = (theta[i+1, j] - 2*theta[i, j] + theta[i-1, j])/dx^2 + ...
        laplacian = (
            (theta[2:, 1:-1] - 2 * theta[1:-1, 1:-1] + theta[:-2, 1:-1]) / dx2 +
            (theta[1:-1, 2:] - 2 * theta[1:-1, 1:-1] + theta[1:-1, :-2]) / dy2
        )
        
        # Mise à jour explicite
        theta[1:-1, 1:-1] += dt_star * laplacian
        
        # Sauvegarde
        if n in save_steps:
            theta_save[save_idx] = theta.copy()
            t_save[save_idx] = n * dt_star
            save_idx += 1
            
    # 5. Conversion en températures physiques
    import torch
    T_save = theta_to_celsius(torch.from_numpy(theta_save)).numpy()
    
    return t_save, X, Y, theta_save, T_save

def main() -> None:
    set_seeds()
    print("Démarrage du solveur DF de référence...")
    
    t0 = time.perf_counter()
    t_save, X, Y, theta_save, T_save = run_fd_solver()
    t1 = time.perf_counter()
    
    print(f"Solveur terminé en {t1 - t0:.3f} s.")
    print(f"Shape de theta_save : {theta_save.shape}")
    
    CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CFG.RESULTS_DIR / "fd_reference.npz"
    
    np.savez_compressed(
        out_path,
        t_save=t_save,
        X=X,
        Y=Y,
        theta=theta_save,
        T=T_save
    )
    print(f"Résultats sauvegardés dans {out_path}")

if __name__ == "__main__":
    main()
