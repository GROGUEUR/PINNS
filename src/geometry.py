"""Objet chaud (disque ou pavé) et condition initiale lissée θ₀_ε (Éq. 2 du sujet).

Sprint 1 — auteur A, relecteur B.

À implémenter (ROADMAP § 3, sprint 1) :
- distance d(x, y) au centre : euclidienne pour le disque, Chebyshev max(|x−cx|, |y−cy|)
  pour le pavé (R est alors le demi-côté) ;
- θ₀_ε(x, y) = ½ [1 − tanh((d − R) / ε)], vectorisé (NumPy et torch), forme (N, 1) ;
- conversions θ ↔ T en °C : T = T_amb + (T_obj − T_amb) · θ.
"""

from typing import Union
import numpy as np
import torch

from src.config import CFG

def theta_to_T(theta: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """Convertit la température adimensionnée θ en température physique T (°C).
    
    Args:
        theta: Température adimensionnée ∈ [0, 1].
        
    Returns:
        Température en °C.
    """
    return CFG.T_AMB + (CFG.T_OBJ - CFG.T_AMB) * theta

def get_distance_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcule la distance au centre selon la forme de l'objet (version NumPy).
    
    Args:
        x: Coordonnée x (N, 1) ou (N,)
        y: Coordonnée y (N, 1) ou (N,)
        
    Returns:
        Distance au centre de l'objet, même forme que l'entrée.
    """
    dx = np.abs(x - CFG.OBJECT_CX)
    dy = np.abs(y - CFG.OBJECT_CY)
    
    if CFG.OBJECT_SHAPE == "disque":
        return np.sqrt(dx**2 + dy**2)
    elif CFG.OBJECT_SHAPE == "pave":
        return np.maximum(dx, dy)
    else:
        raise ValueError(f"Forme inconnue: {CFG.OBJECT_SHAPE}")

def theta_0_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Condition initiale lissée (version NumPy).
    
    Args:
        x: Coordonnée x (N, 1) ou autre forme valide.
        y: Coordonnée y (N, 1) ou autre forme valide.
        
    Returns:
        θ₀_ε, même forme que l'entrée.
    """
    d = get_distance_np(x, y)
    return 0.5 * (1.0 - np.tanh((d - CFG.OBJECT_RADIUS) / CFG.EPS_IC))

def get_distance_pt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calcule la distance au centre selon la forme de l'objet (version PyTorch).
    
    Args:
        x: Coordonnée x (N, 1)
        y: Coordonnée y (N, 1)
        
    Returns:
        Distance au centre de l'objet, forme (N, 1).
    """
    dx = torch.abs(x - CFG.OBJECT_CX)
    dy = torch.abs(y - CFG.OBJECT_CY)
    
    if CFG.OBJECT_SHAPE == "disque":
        return torch.sqrt(dx**2 + dy**2)
    elif CFG.OBJECT_SHAPE == "pave":
        return torch.maximum(dx, dy)
    else:
        raise ValueError(f"Forme inconnue: {CFG.OBJECT_SHAPE}")

def theta_0_pt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Condition initiale lissée (version PyTorch).
    
    Args:
        x: Coordonnée x (N, 1).
        y: Coordonnée y (N, 1).
        
    Returns:
        θ₀_ε, forme (N, 1).
    """
    d = get_distance_pt(x, y)
    return 0.5 * (1.0 - torch.tanh((d - CFG.OBJECT_RADIUS) / CFG.EPS_IC))
