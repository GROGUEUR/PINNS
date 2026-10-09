"""Figures : cartes θ̂ / θ_DF / |erreur| côte à côte, courbes de pertes, erreur en temps.

Sprint 2 — auteur B, relecteur A.

À implémenter (ROADMAP § 3, sprint 2) : matplotlib uniquement, échelle de couleur fixe
entre les panneaux pour que les cartes soient comparables.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.config import CFG

def plot_heatmaps(theta_pred: np.ndarray, theta_ref: np.ndarray, t_star: float, save_path: Path = None):
    """Génère 3 cartes de chaleur côte à côte : PINN, DF, et Erreur absolue.
    
    L'échelle de couleur est fixée entre 0 et 1 pour les champs, 
    et adaptée pour l'erreur.
    
    Args:
        theta_pred: Prédictions du modèle (FD_N, FD_N).
        theta_ref: Référence DF (FD_N, FD_N).
        t_star: L'instant adimensionné t* affiché.
        save_path: Chemin de sauvegarde optionnel.
    """
    error = np.abs(theta_pred - theta_ref)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Prédiction PINN
    im0 = axes[0].imshow(theta_pred, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1, cmap="inferno")
    axes[0].set_title(f"PINN θ (t*={t_star:.3f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])
    
    # 2. Référence DF
    im1 = axes[1].imshow(theta_ref, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1, cmap="inferno")
    axes[1].set_title(f"DF θ (t*={t_star:.3f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])
    
    # 3. Erreur absolue
    im2 = axes[2].imshow(error, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[2].set_title("Erreur absolue |θ_PINN - θ_DF|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_loss_curves(history: dict, save_path: Path = None):
    """Trace l'évolution des pertes au cours de l'entraînement.
    
    Args:
        history: Dictionnaire contenant les listes de pertes.
                 (ex: {"total": [...], "ic": [...], "bc": [...], "res": [...]})
        save_path: Chemin de sauvegarde optionnel.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for key, values in history.items():
        if len(values) > 0:
            x_axis = np.arange(1, len(values) + 1)
            ax.semilogy(x_axis, values, label=f"Loss {key}")
            
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Pertes (log scale)")
    ax.set_title("Courbes d'entraînement du PINN")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_error_vs_time(t_save: np.ndarray, rel_l2_errors: np.ndarray, save_path: Path = None):
    """Trace l'évolution de l'erreur L2 relative en fonction du temps t*.
    
    Args:
        t_save: Instants sauvegardés.
        rel_l2_errors: Erreurs L2 relatives à chaque instant.
        save_path: Chemin de sauvegarde optionnel.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t_save, rel_l2_errors * 100, "o-", color="firebrick")  # en pourcentage
    
    ax.set_xlabel("t* (Temps adimensionné)")
    ax.set_ylabel("Erreur L2 relative (%)")
    ax.set_title("Évolution de l'erreur au cours du temps")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()
