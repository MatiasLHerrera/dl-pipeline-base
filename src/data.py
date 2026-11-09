"""Carga del dataset, partición train/val/test estratificada, escalado y DataLoaders."""
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
TARGET = "target"


def _to_dataset(X, y) -> TensorDataset:
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))


def get_dataloaders(cfg):
    df = pd.read_csv(cfg.data_path)
    X, y = df[FEATURES].values, df[TARGET].values

    # 1) Separamos test (nunca se toca hasta el final)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    # 2) Del resto separamos validación
    val_frac = cfg.val_size / (1 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=cfg.seed
    )

    # El scaler se ajusta SOLO con train para evitar data leakage
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))

    g = torch.Generator().manual_seed(cfg.seed)
    loaders = {
        "train": DataLoader(_to_dataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True, generator=g),
        "val": DataLoader(_to_dataset(X_val, y_val), batch_size=cfg.batch_size),
        "test": DataLoader(_to_dataset(X_test, y_test), batch_size=cfg.batch_size),
    }
    meta = {
        "n_features": X_train.shape[1],
        "n_classes": int(len(set(y))),
        "sizes": {k: len(v.dataset) for k, v in loaders.items()},
    }
    return loaders, meta
