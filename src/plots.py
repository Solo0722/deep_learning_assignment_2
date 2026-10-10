"""All figures.  python -m src.plots --results-dir results/mnist"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import FACTORS


def factor_plots(df, summ, figdir):
    present = set(df.config)
    for factor, names in FACTORS.items():
        names = [n for n in names if n in present]
        if len(names) < 2:
            continue
        s = summ.set_index("config").loc[names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for i, n in enumerate(names):
            d = df[df.config == n]
            ax[0].scatter([i] * len(d), 100 * d.test_acc, s=14, color="gray", alpha=0.6, zorder=2)
        ax[0].errorbar(x, s.test_acc_mean, yerr=s.test_acc_std, fmt="o-", capsize=4, zorder=3)
        ax[0].set_ylabel("Test accuracy (%)  [mean ± std, dots = seeds]")
        ax[0].set_title(f"{factor}: test accuracy")
        ax[1].bar(x, s.gap_mean)
        ax[1].set_ylabel("Train − val accuracy (pp)")
        ax[1].set_title(f"{factor}: overfitting gap")
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels(names, rotation=30, ha="right")
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / f"factor_{factor}.png", dpi=150)
        plt.close(fig)


def params_scatter(summ, figdir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    owner = {}
    for f, names in FACTORS.items():
        for n in names:
            owner.setdefault(n, f if n != "baseline" else "baseline")
    colors = {f: c for f, c in zip(["baseline"] + list(FACTORS), plt.cm.tab10.colors)}
    for _, r in summ.iterrows():
        f = owner.get(r.config, "baseline")
        ax.errorbar(r.params, r.test_acc_mean, yerr=r.test_acc_std, fmt="o", capsize=3, color=colors[f],
                    label=f)
        ax.annotate(r.config, (r.params, r.test_acc_mean), fontsize=7, xytext=(3, 3), textcoords="offset points")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs parameter count: design effect or just capacity?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "accuracy_vs_params.png", dpi=150)
    plt.close(fig)


def baseline_curves(out, figdir):
    files = sorted((out / "curves").glob("baseline_s*.json"))
    if not files:
        return
    cs = [json.loads(f.read_text()) for f in files]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for key, style in [("train", "--"), ("val", "-")]:
        ax[0].plot(np.mean([c[f"{key}_loss"] for c in cs], 0), style, label=key)
        ax[1].plot(100 * np.mean([c[f"{key}_acc"] for c in cs], 0), style, label=key)
    ax[0].set_title("Baseline loss (mean over seeds)")
    ax[1].set_title("Baseline accuracy % (mean over seeds)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "baseline_curves.png", dpi=150)
    plt.close(fig)


def plot_cm(cm, title, path):
    shown = cm.copy().astype(float)
    np.fill_diagonal(shown, 0)  # colour only the errors, otherwise the diagonal hides everything
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(shown, cmap="Reds")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="black" if i != j else "gray")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title + "\n(counts summed over seeds; colour = errors only)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_plots(out, summ, figdir):
    # best / worst are chosen by mean VALIDATION accuracy, not test (selecting on test = leakage)
    best = summ.loc[summ.val_acc_mean.idxmax(), "config"]
    worst = summ.loc[summ.val_acc_mean.idxmin(), "config"]
    for tag, name in [("baseline", "baseline"), ("best", best), ("worst", worst)]:
        files = sorted((out / "confusion").glob(f"{name}_s*.npy"))
        if files:
            plot_cm(sum(np.load(f) for f in files), f"{tag}: {name}", figdir / f"confusion_{tag}_{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/mnist")
    a = ap.parse_args()
    out = Path(a.results_dir)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    df = pd.read_csv(out / "runs.csv")
    summ = pd.read_csv(out / "tables" / "summary.csv")  # run src.aggregate first
    factor_plots(df, summ, figdir)
    params_scatter(summ, figdir)
    baseline_curves(out, figdir)
    confusion_plots(out, summ, figdir)
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
