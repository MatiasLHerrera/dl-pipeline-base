"""Punto de entrada del pipeline: python src/train.py [--lr 1e-3 --epochs 100 ...]"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config, get_device, set_seed  # noqa: E402
from data import get_dataloaders  # noqa: E402
from engine import evaluate, train_one_epoch  # noqa: E402
from model import MLPClassifier  # noqa: E402


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Pipeline base de entrenamiento y validación")
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--seed", type=int, default=cfg.seed)
    a = p.parse_args()
    cfg.learning_rate, cfg.epochs, cfg.batch_size, cfg.seed = a.lr, a.epochs, a.batch_size, a.seed
    return cfg


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], label="train")
    ax1.plot(ep, [h["val_loss"] for h in history], label="val")
    ax1.set(title="Pérdida (CrossEntropy)", xlabel="Época", ylabel="Loss")
    ax2.plot(ep, [h["train_acc"] for h in history], label="train")
    ax2.plot(ep, [h["val_acc"] for h in history], label="val")
    ax2.set(title="Accuracy", xlabel="Época", ylabel="Accuracy", ylim=(0, 1.02))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"PyTorch {torch.__version__} | dispositivo: {device}")

    loaders, meta = get_dataloaders(cfg)
    print(f"Particiones: {meta['sizes']}")

    model = MLPClassifier(meta["n_features"], meta["n_classes"], cfg.hidden_dims, cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    print(model)

    history, best_val_loss, best_path = [], float("inf"), cfg.output_dir / "best_model.pt"
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc, va_f1 = evaluate(model, loaders["val"], criterion, device)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1))

        if va_loss < best_val_loss:  # checkpoint del mejor modelo según validación
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Época {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

    # Evaluación final en test con el mejor checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, te_f1 = evaluate(model, loaders["test"], criterion, device)
    print(f"\nTest (mejor checkpoint) | loss {te_loss:.4f} acc {te_acc:.3f} f1 {te_f1:.3f}")

    with open(cfg.output_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    summary = dict(torch_version=torch.__version__, device=str(device), learning_rate=cfg.learning_rate,
                   epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed, sizes=meta["sizes"],
                   best_val_loss=best_val_loss, test_loss=te_loss, test_acc=te_acc, test_f1=te_f1)
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_history(history, cfg.output_dir / "curvas.png")
    print(f"Resultados guardados en {cfg.output_dir}")


if __name__ == "__main__":
    main()
