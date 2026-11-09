"""Ciclos de entrenamiento y evaluación."""
import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)          # mismo dispositivo que el modelo

        optimizer.zero_grad()                      # 1) reiniciar gradientes (se acumulan por defecto)
        logits = model(X)                          # 2) forward pass
        loss = criterion(logits, y)                # 3) cálculo de la pérdida
        loss.backward()                            # 4) backward pass (torch.autograd)
        optimizer.step()                           # 5) actualización de pesos (Adam)

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += X.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Pérdida, Accuracy y F1 macro sobre datos no vistos en el entrenamiento."""
    model.eval()                                   # desactiva dropout
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)
        n += X.size(0)
        y_true.append(y.cpu())
        y_pred.append(logits.argmax(dim=1).cpu())
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    acc = (y_true == y_pred).float().mean().item()
    f1 = f1_score(y_true.numpy(), y_pred.numpy(), average="macro")
    return total_loss / n, acc, f1
