"""Constantes physiques et hyperparamètres du projet : source unique de vérité.

Règle du dépôt (AGENTS.md § 1, règle 4) : aucune valeur numérique n'est écrite
« en dur » ailleurs que dans ce fichier. Les autres modules font
``from src.config import CFG`` et lisent ses champs.

Conventions :
- une grandeur physique porte son unité en commentaire de fin de ligne ;
- « sans dim » signifie adimensionné (AGENTS.md § 2, docs/physics.md) ;
- ``Config`` est figée (``frozen=True``) : on ne modifie jamais ``CFG`` en place.
  Pour une variante (ablation, test), on écrit ``dataclasses.replace(CFG, EPS_IC=0.02)``,
  ce qui crée une nouvelle instance et relance les garde-fous de ``__post_init__``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    """Toutes les constantes du projet, groupées par thème.

    Les valeurs reprennent la table des décisions verrouillées (AGENTS.md § 6).
    Toute modification doit y être reportée et validée par le binôme.
    """

    # ------------------------------------------------------------------
    # 1. Physique du sujet (unités SI ; sujet § 1 et § 2)
    # ------------------------------------------------------------------
    L: float = 1.0          # m    : côté de la pièce carrée (Lx = Ly)
    ALPHA: float = 2e-5     # m²/s : diffusivité thermique de l'air (Éq. 1)
    T_AMB: float = 20.0     # °C   : température ambiante, murs et pièce (Éq. 2, 3)
    T_OBJ: float = 80.0     # °C   : température initiale de l'objet (Éq. 2)

    # ------------------------------------------------------------------
    # 2. Adimensionnement (docs/physics.md)
    # ------------------------------------------------------------------
    # Horizon adimensionné t*_max = α·t_max/L². À t* = 0.1, θ_max ≈ 0.017 :
    # la pièce est presque revenue à l'équilibre (mesure DF, ROADMAP § 0).
    # Le réseau reçoit τ = t*/t*_max ∈ [0, 1] ; ce facteur apparaît donc dans le résidu.
    T_STAR_MAX: float = 0.1  # sans dim

    # ------------------------------------------------------------------
    # 3. Objet chaud et condition initiale lissée (Éq. 2)
    # ------------------------------------------------------------------
    OBJECT_SHAPE: str = "disque"  # "disque" ou "pave" (défaut provisoire, AGENTS.md § 10)
    OBJECT_RADIUS: float = 0.1    # sans dim (fraction de L) : rayon du disque ou demi-côté du pavé
    OBJECT_CX: float = 0.5        # sans dim : centre de l'objet, milieu de la pièce
    OBJECT_CY: float = 0.5        # sans dim
    # Largeur de la transition tanh de θ₀_ε = ½[1 − tanh((d − R)/ε)]. Un échelon n'est
    # pas représentable par un MLP lisse ; ε est un compromis fidélité / entraînabilité.
    EPS_IC: float = 0.01          # sans dim, à ajuster entre 0.005 et 0.02

    # ------------------------------------------------------------------
    # 4. Points de collocation (travail 1 du sujet)
    # ------------------------------------------------------------------
    N_RES: int = 20_000  # points intérieurs (x, y, τ) où l'on évalue le résidu de l'EDP
    N_IC: int = 5_000    # points à τ = 0
    N_BC: int = 4_000    # points sur les 4 murs, pour tout τ
    # τ = u^TAU_EXPONENT avec u ~ U[0, 1] : concentre les points près de τ = 0,
    # là où se joue toute la dynamique (θ_max passe de 1 à 0.39 en 5 % du temps).
    TAU_EXPONENT: float = 2.0
    # Densification de l'IC autour du bord de l'objet, là où θ₀_ε varie vite :
    # une part IC_EDGE_FRACTION des N_IC points est tirée dans la bande |d − R| ≤ ic_edge_band.
    IC_EDGE_FRACTION: float = 0.5
    IC_EDGE_BAND_EPS: float = 3.0  # demi-largeur de la bande en multiples de EPS_IC (tanh(3) ≈ 0.995)

    # ------------------------------------------------------------------
    # 5. Réseau et perte (travail 2 du sujet, Éq. 4)
    # ------------------------------------------------------------------
    HIDDEN_LAYERS: tuple[int, ...] = (64, 64, 64, 64)  # MLP 3 → [64]×4 → 1, tanh, ~13k paramètres
    # Poids statiques de la perte L = w_IC·L_IC + w_BC·L_BC + w_res·L_res.
    # Valent 1 pour la baseline ; remplacés par les poids dynamiques au sprint 3.
    W_IC: float = 1.0
    W_BC: float = 1.0
    W_RES: float = 1.0

    # ------------------------------------------------------------------
    # 6. Entraînement hybride : Adam puis L-BFGS (travail 3 du sujet)
    # ------------------------------------------------------------------
    ADAM_LR: float = 1e-3
    ADAM_ITERS: int = 20_000
    ADAM_LR_DECAY_RATE: float = 0.9   # lr ← lr × 0.9 tous les ADAM_LR_DECAY_STEPS pas
    ADAM_LR_DECAY_STEPS: int = 2_000  # soit lr_final ≈ 0.9¹⁰ ≈ 0.35 × lr_initial
    LBFGS_LR: float = 1.0
    LBFGS_MAX_ITER: int = 3_000       # valeur de départ, à ajuster au sprint 3
    LBFGS_HISTORY_SIZE: int = 50
    LBFGS_LINE_SEARCH: str = "strong_wolfe"
    LOG_EVERY: int = 500              # seuls les logs font des .item() (AGENTS.md § 5)

    # ------------------------------------------------------------------
    # 7. Techniques avancées pour la raideur de l'IC (sprint 3)
    # ------------------------------------------------------------------
    # RAD (Wu et al. 2023) : p ∝ |r|^k / E[|r|^k] + c, défauts recommandés k = 1, c = 1.
    RAD_K: float = 1.0
    RAD_C: float = 1.0
    RAD_EVERY: int = 2_000        # itérations Adam entre deux rééchantillonnages
    RAD_POOL_SIZE: int = 100_000  # candidats parmi lesquels on retire N_RES points
    # Poids dynamiques par normes de gradients (Wang et al. 2023, Expert's Guide) :
    # λ̂ᵢ = Σⱼ‖∇Lⱼ‖ / ‖∇Lᵢ‖, lissé par λ ← 0.9·λ + 0.1·λ̂.
    WEIGHT_UPDATE_EVERY: int = 1_000
    WEIGHT_MOMENTUM: float = 0.9

    # ------------------------------------------------------------------
    # 8. Solveur de référence FTCS et grille des métriques (travail 4 du sujet)
    # ------------------------------------------------------------------
    FD_N: int = 101             # nœuds par direction, donc dx = 1/100
    FD_CFL_FACTOR: float = 0.2  # dt* = 0.2·dx² ; le schéma explicite 2D est stable ssi dt* ≤ dx²/4
    FD_N_SAVE: int = 51         # instants sauvegardés ; grille des métriques = 101 × 101 × 51

    # ------------------------------------------------------------------
    # 9. PINN paramétrique et démo Gradio (sprint 4)
    # ------------------------------------------------------------------
    PARAM_R_MIN: float = 0.05       # sans dim : plage du rayon tiré au hasard à l'entraînement
    PARAM_R_MAX: float = 0.2
    PARAM_CENTER_MIN: float = 0.3   # sans dim : plage de cx et cy
    PARAM_CENTER_MAX: float = 0.7

    # ------------------------------------------------------------------
    # 10. Reproductibilité, matériel, chemins
    # ------------------------------------------------------------------
    SEED: int = 0
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # le code doit tourner sur CPU
    CHECKPOINT_DIR: Path = Path("checkpoints")
    RESULTS_DIR: Path = Path("results")

    # ------------------------------------------------------------------
    # Grandeurs dérivées : calculées, jamais recopiées à la main
    # ------------------------------------------------------------------
    @property
    def t_char_s(self) -> float:
        """Temps caractéristique de diffusion L²/α, en secondes (50 000 s ≈ 13,9 h)."""
        return self.L**2 / self.ALPHA

    @property
    def t_max_s(self) -> float:
        """Horizon physique t_max = t*_max · L²/α, en secondes (5 000 s ≈ 1,4 h)."""
        return self.T_STAR_MAX * self.t_char_s

    @property
    def delta_t(self) -> float:
        """Écart T_obj − T_amb, en °C (60 °C). Sert aux conversions θ ↔ T."""
        return self.T_OBJ - self.T_AMB

    @property
    def ic_edge_band(self) -> float:
        """Demi-largeur (sans dim) de la bande de densification de l'IC : IC_EDGE_BAND_EPS · ε."""
        return self.IC_EDGE_BAND_EPS * self.EPS_IC

    @property
    def fd_dx(self) -> float:
        """Pas d'espace adimensionné de la grille DF (1/100)."""
        return 1.0 / (self.FD_N - 1)

    @property
    def fd_dt_star(self) -> float:
        """Pas de temps adimensionné t* du schéma FTCS : FD_CFL_FACTOR · dx²."""
        return self.FD_CFL_FACTOR * self.fd_dx**2

    def __post_init__(self) -> None:
        """Garde-fous contre les pièges connus (AGENTS.md § 7). Lève ValueError."""
        if self.OBJECT_SHAPE not in ("disque", "pave"):
            raise ValueError(f"OBJECT_SHAPE doit valoir 'disque' ou 'pave', reçu {self.OBJECT_SHAPE!r}")

        # Piège § 7 : le schéma explicite FTCS 2D diverge si dt* > dx²/4.
        if self.FD_CFL_FACTOR > 0.25:
            raise ValueError(f"FD_CFL_FACTOR = {self.FD_CFL_FACTOR} > 0.25 : schéma FTCS instable")

        # L'échantillonnage BC répartit les points à parts égales sur les 4 murs.
        if self.N_BC % 4 != 0:
            raise ValueError(f"N_BC = {self.N_BC} doit être un multiple de 4 (4 murs)")
        if not 0.0 <= self.IC_EDGE_FRACTION <= 1.0:
            raise ValueError(f"IC_EDGE_FRACTION = {self.IC_EDGE_FRACTION} doit être dans [0, 1]")

        # L'objet doit tenir dans la pièce, sinon l'IC (θ = 1) contredit la BC (θ = 0).
        r = self.OBJECT_RADIUS
        inside_x = r < self.OBJECT_CX < 1.0 - r
        inside_y = r < self.OBJECT_CY < 1.0 - r
        if not (inside_x and inside_y):
            raise ValueError("L'objet déborde de la pièce : vérifier OBJECT_RADIUS, OBJECT_CX, OBJECT_CY")


def set_seeds(seed: int = Config.SEED) -> None:
    """Fixe toutes les graines aléatoires pour rendre une exécution reproductible.

    À appeler au début de chaque script (entraînement, DF, évaluation).

    Args:
        seed: graine entière commune à ``random``, NumPy et PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # couvre aussi les générateurs CUDA s'il y a un GPU


# Instance unique importée par tout le projet.
CFG = Config()
