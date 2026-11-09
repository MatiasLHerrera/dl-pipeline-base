"""Configuración central del experimento: hiperparámetros, rutas, dispositivo y semillas."""
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Datos
    data_path: Path = ROOT / "data" / "iris.csv"
    output_dir: Path = ROOT / "outputs"
    val_size: float = 0.2
    test_size: float = 0.2

    # Modelo
    hidden_dims: tuple = (16, 16)
    dropout: float = 0.1

    # Entrenamiento
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 100
    weight_decay: float = 0.0

    seed: int = 42
    extra: dict = field(default_factory=dict)


def get_device() -> torch.device:
    """Detecta automáticamente el mejor dispositivo disponible: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Fija todas las fuentes de aleatoriedad para que el experimento sea reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
