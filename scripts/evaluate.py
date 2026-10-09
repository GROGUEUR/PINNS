"""Évaluation d'un checkpoint : métriques + figures dans results/.

Sprint 2 — auteur B, relecteur A.
Usage prévu : python -m scripts.evaluate --ckpt checkpoints/<nom>.pt
"""

import argparse
import numpy as np
import torch
import time
from pathlib import Path

from src.config import CFG, set_seeds
from src.metrics import mse, relative_l2_error, max_error_celsius
from src.viz import plot_heatmaps, plot_error_vs_time

try:
    from src.train import load_checkpoint
except ImportError:
    load_checkpoint = None

def load_reference_data():
    """Charge les données du solveur DF."""
    ref_path = CFG.RESULTS_DIR / "fd_reference.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"Fichier de référence introuvable: {ref_path}")
    
    data = np.load(ref_path)
    return data["t_save"], data["X"], data["Y"], data["theta"], data["T"]

def generate_predictions(model, t_save, X, Y):
    """Génère les prédictions du modèle sur la grille (X, Y) pour les instants t_save.
    
    Args:
        model: Modèle PyTorch (PINN).
        t_save: Instants de temps (FD_N_SAVE,).
        X, Y: Grilles spatiales (FD_N, FD_N).
        
    Returns:
        theta_pred: Tensor NumPy des prédictions (FD_N_SAVE, FD_N, FD_N).
    """
    FD_N = CFG.FD_N
    N_SAVE = len(t_save)
    theta_pred = np.zeros((N_SAVE, FD_N, FD_N))
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(t_save):
            t_flat = np.full_like(x_flat, fill_value=t)
            
            # Forme (N, 3) pour le modèle : x, y, tau (car tau = t* / t*_max)
            # Attention, le solveur enregistre en t*, il faut convertir en tau pour le réseau !
            tau_flat = t_flat / CFG.T_STAR_MAX
            
            inputs = np.stack([x_flat, y_flat, tau_flat], axis=-1)
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=CFG.DEVICE)
            
            preds = model(inputs_tensor)
            
            theta_pred[i] = preds.cpu().numpy().reshape(FD_N, FD_N)
            
    return theta_pred

def run_evaluation(ckpt_path: Path):
    if load_checkpoint is None:
        raise ImportError("La fonction load_checkpoint n'est pas encore implémentée dans src.train (Auteur A).")
        
    set_seeds()
    
    print(f"Chargement du checkpoint : {ckpt_path}")
    model, cfg, checkpoint = load_checkpoint(ckpt_path)
    model.eval()
    
    print("Chargement des données de référence DF...")
    t_save, X, Y, theta_ref, T_ref = load_reference_data()
    
    print("Génération des prédictions du PINN...")
    t0 = time.perf_counter()
    theta_pred = generate_predictions(model, t_save, X, Y)
    t1 = time.perf_counter()
    print(f"Prédictions générées en {t1 - t0:.3f} s.")
    
    # Calcul des métriques globales
    mse_val = mse(theta_pred, theta_ref)
    rel_l2 = relative_l2_error(theta_pred, theta_ref)
    max_err_celsius = max_error_celsius(theta_pred, theta_ref)
    
    print(f"\nMétriques globales :")
    print(f"  MSE : {mse_val:.2e}")
    print(f"  Erreur L2 relative : {rel_l2:.2e}")
    print(f"  Erreur Max (°C) : {max_err_celsius:.2f} °C")
    
    # Calcul de l'erreur L2 relative pour chaque instant
    l2_errors_time = []
    for i in range(len(t_save)):
        err = relative_l2_error(theta_pred[i], theta_ref[i])
        l2_errors_time.append(err)
    l2_errors_time = np.array(l2_errors_time)
    
    # Création du rapport
    report_name = ckpt_path.stem
    report_path = CFG.RESULTS_DIR / f"{report_name}.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Évaluation du modèle : {report_name}\n\n")
        f.write(f"- **MSE** : {mse_val:.2e}\n")
        f.write(f"- **Erreur L2 relative globale** : {rel_l2:.2e}\n")
        f.write(f"- **Erreur absolue maximale** : {max_err_celsius:.2f} °C\n")
        f.write(f"- **Temps d'inférence (grille 101x101x51)** : {t1 - t0:.3f} s\n")
        
    print(f"Rapport sauvegardé dans {report_path}")
    
    # Génération des figures
    fig_error_path = CFG.RESULTS_DIR / f"{report_name}_error_vs_time.png"
    plot_error_vs_time(t_save, l2_errors_time, save_path=fig_error_path)
    
    # Carte d'erreur à la fin de la simulation (t* = T_STAR_MAX)
    fig_heatmap_path = CFG.RESULTS_DIR / f"{report_name}_heatmap_final.png"
    plot_heatmaps(theta_pred[-1], theta_ref[-1], t_save[-1], save_path=fig_heatmap_path)
    
    # Courbe des pertes d'entraînement
    if "history" in checkpoint:
        from src.viz import plot_loss_curves
        history_tensor = checkpoint["history"].cpu().numpy()
        history_dict = {
            "Total": history_tensor[:, 0],
            "IC": history_tensor[:, 1],
            "BC": history_tensor[:, 2],
            "Res": history_tensor[:, 3]
        }
        fig_loss_path = CFG.RESULTS_DIR / f"{report_name}_loss_curves.png"
        plot_loss_curves(history_dict, save_path=fig_loss_path)
    
    print(f"Figures générées dans {CFG.RESULTS_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue un checkpoint PINN vis-à-vis du solveur DF.")
    parser.add_argument("--ckpt", type=str, required=True, help="Chemin vers le fichier checkpoint (.pt)")
    args = parser.parse_args()
    
    ckpt_path = Path(args.ckpt)
    run_evaluation(ckpt_path)
