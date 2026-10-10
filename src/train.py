"""One function does everything for one (config, seed): build, train, select best epoch on
VALIDATION, evaluate on test."""
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .model import LeNetVariant, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, x, y, bs=2000):
    model.eval()
    loss, preds = 0.0, []
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        preds.append(out.argmax(1))
    preds = torch.cat(preds)
    return loss / len(x), (preds == y).float().mean().item(), preds


def train_and_eval(cfg, seed, data, hp, device):
    set_seed(seed)
    model = LeNetVariant(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    n, bs = len(xtr), hp["batch_size"]
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    curves = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val, best_epoch, best_state = -1.0, 0, None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=device, generator=g)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
            correct += (out.argmax(1) == yb).sum()
        val_loss, val_acc, _ = evaluate(model, xva, yva)
        curves["train_loss"].append(loss_sum.item() / n)
        curves["train_acc"].append(correct.item() / n)
        curves["val_loss"].append(val_loss)
        curves["val_acc"].append(val_acc)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0

    model.load_state_dict(best_state)  # best epoch chosen on validation only
    _, train_acc, _ = evaluate(model, xtr, ytr)  # eval mode: no dropout / BN batch stats
    _, val_acc, _ = evaluate(model, xva, yva)
    test_loss, test_acc, pred = evaluate(model, xte, yte)

    y_true, y_pred = yte.cpu().numpy(), pred.cpu().numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    row = {
        "config": cfg["name"], "seed": seed, "params": count_params(model),
        "train_time_s": train_time, "best_epoch": best_epoch,
        "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "test_loss": test_loss,
        "precision_macro": prec, "recall_macro": rec, "f1_macro": f1,
    }
    return row, curves, cm
