"""Run experiments. Resumable: (config, seed) pairs already in runs.csv are skipped.

    python -m src.run_all --smoke                     # 1-minute pipeline check (separate results dir)
    python -m src.run_all                             # all factors, 5 seeds, MNIST
    python -m src.run_all --factors width activation  # subset
    python -m src.run_all --factors combined          # after you filled in COMBINED_NAMES
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import N_VAL, SPLIT_SEED, load_data
from .experiments import CONFIGS, FACTORS
from .train import train_and_eval

FIELDS = ["config", "seed", "params", "train_time_s", "best_epoch", "train_acc", "val_acc",
          "test_acc", "test_loss", "precision_macro", "recall_macro", "f1_macro"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--factors", nargs="+", default=["all"])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    if a.smoke:
        a.seeds, a.epochs = 1, 2
        a.results_dir = a.results_dir or "results/_smoke"
        shutil.rmtree(a.results_dir, ignore_errors=True)
    out = Path(a.results_dir or f"results/{a.dataset}")
    (out / "curves").mkdir(parents=True, exist_ok=True)
    (out / "confusion").mkdir(parents=True, exist_ok=True)

    # Hyperparameters must be identical across every run in a results dir, otherwise comparisons are invalid.
    hp = {"dataset": a.dataset, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
          "optimizer": "adam", "n_val": N_VAL, "split_seed": SPLIT_SEED}
    hp_path = out / "hparams.json"
    if hp_path.exists() and json.loads(hp_path.read_text()) != hp:
        raise SystemExit(f"Hyperparameters differ from those in {hp_path}. Use a new --results-dir "
                         f"(mixing settings would make results incomparable).")
    hp_path.write_text(json.dumps(hp, indent=2))
    (out / "configs.json").write_text(json.dumps(CONFIGS, indent=2))

    keys = list(FACTORS) if a.factors == ["all"] else a.factors
    names = []
    for k in keys:
        for n in FACTORS[k]:
            if n not in names:
                names.append(n)

    runs_csv = out / "runs.csv"
    done = set()
    if runs_csv.exists():
        with open(runs_csv) as f:
            done = {(r["config"], int(r["seed"])) for r in csv.DictReader(f)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  dataset={a.dataset}  configs={len(names)}  seeds={a.seeds}  epochs={a.epochs}")
    data = load_data(a.dataset, a.data_dir, device, train_subset=3000 if a.smoke else None)

    total = len(names) * a.seeds
    k = 0
    for seed in range(a.seeds):  # seeds outer: if interrupted you still have complete seeds
        for name in names:
            k += 1
            if (name, seed) in done:
                continue
            row, curves, cm = train_and_eval(CONFIGS[name], seed, data, hp, device)
            new = not runs_csv.exists()
            with open(runs_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            (out / "curves" / f"{name}_s{seed}.json").write_text(json.dumps(curves))
            np.save(out / "confusion" / f"{name}_s{seed}.npy", cm)
            print(f"[{k}/{total}] {name:<16} seed={seed} params={row['params']:>7} "
                  f"val={row['val_acc']*100:.2f} test={row['test_acc']*100:.2f} "
                  f"f1={row['f1_macro']*100:.2f} ({row['train_time_s']:.0f}s)", flush=True)
    print("done ->", runs_csv)


if __name__ == "__main__":
    main()
