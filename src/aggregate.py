"""Aggregate runs.csv -> mean +/- std per config, Welch t-test vs baseline, markdown tables per factor.

    python -m src.aggregate --results-dir results/mnist
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .experiments import FACTORS

METRICS = ["train_acc", "val_acc", "test_acc", "precision_macro", "recall_macro", "f1_macro"]


def summarize(df):
    base = df[df.config == "baseline"]
    comps = [c for c in df.config.unique() if c != "baseline"]
    alpha = 0.05 / max(len(comps), 1) 
    rows = []
    for c, d in df.groupby("config", sort=False):
        r = {"config": c, "n_seeds": len(d), "params": int(d.params.iloc[0]), "time_s": d.train_time_s.mean()}
        for m in METRICS:
            r[m + "_mean"] = 100 * d[m].mean()
            r[m + "_std"] = 100 * d[m].std(ddof=1) if len(d) > 1 else np.nan
        r["gap_mean"] = 100 * (d.train_acc - d.val_acc).mean()
        r["delta_test_acc"], r["p_value"], r["verdict"] = np.nan, np.nan, "-"
        if c != "baseline" and len(d) > 1 and len(base) > 1:
            _, p = stats.ttest_ind(d.test_acc, base.test_acc, equal_var=False)
            r["delta_test_acc"] = r["test_acc_mean"] - 100 * base.test_acc.mean()
            r["p_value"] = p
            if np.isnan(p):
                r["verdict"] = "n/a"
            elif p < alpha:
                r["verdict"] = "better" if r["delta_test_acc"] > 0 else "worse"
            else:
                r["verdict"] = "no evidence of difference"
        rows.append(r)
    return pd.DataFrame(rows), alpha


def pm(r, m):
    return f"{r[m + '_mean']:.2f} ± {r[m + '_std']:.2f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/mnist")
    a = ap.parse_args()
    out = Path(a.results_dir)
    df = pd.read_csv(out / "runs.csv")
    (out / "tables").mkdir(exist_ok=True)

    summ, alpha = summarize(df)
    summ.to_csv(out / "tables" / "summary.csv", index=False)

    headers = ["config", "params", "test acc %", "precision %", "recall %", "F1 %",
               "train-val gap", "Δ test acc", "p", "verdict"]
    for factor, names in FACTORS.items():
        rows = []
        for n in names:
            m = summ[summ.config == n]
            if m.empty:
                continue
            r = m.iloc[0]
            rows.append([n, f"{r.params:,}", pm(r, "test_acc"), pm(r, "precision_macro"),
                         pm(r, "recall_macro"), pm(r, "f1_macro"), f"{r.gap_mean:.2f}",
                         "-" if np.isnan(r.delta_test_acc) else f"{r.delta_test_acc:+.2f}",
                         "-" if np.isnan(r.p_value) else f"{r.p_value:.3g}", r.verdict])
        if len(rows) > 1:
            text = md_table(headers, rows)
            (out / "tables" / f"factor_{factor}.md").write_text(text + "\n")
            print(f"\n### {factor}\n{text}")
    print(f"\nn seeds per config: {sorted(summ.n_seeds.unique())}. Welch t-test vs baseline, "
          f"Bonferroni alpha = {alpha:.4f}. 'no evidence of difference' != 'equal'.")


if __name__ == "__main__":
    main()
