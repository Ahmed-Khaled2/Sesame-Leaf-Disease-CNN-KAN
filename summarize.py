"""
summarize.py - collect ALL finished runs into one Excel file: outputs/all_runs.xlsx

Sheets:
  runs           one row per run: settings + mean/std of every score
  folds          one row per run and fold (all fold numbers, with the run settings)
  phase1_choice  phase 1 runs ranked by VALIDATION macro-F1 (preprocessing, backbone)
  phase2_choice  phase 2 runs ranked by VALIDATION macro-F1 (best KAN and best MLP config)
  by_config      phase 3 style table: every setting averaged over ALL its seeds and folds
                 (this is the table for the paper)
  comparisons    paired head-vs-head comparisons on the TEST folds (same data, same seeds/folds):
                 mean difference, 95% CI and the Nadeau-Bengio corrected t-test, plus an
                 equivalence test (TOST, same correction): are the two heads equal within
                 +-2 accuracy points (+-0.02 F1)? and the smallest margin that would pass
  other_comparisons  same head, paired: multi vs multi_syn (does synthetic data help?) and
                 frozen vs layer4 unfrozen
  kan_check      spline checks of every KAN run
  kan_knockout   test scores of every KAN run with the spline part / base part switched off
  speed          measured latency (from bench_speed.py), if it has been run
  kan_curves     spline share per KAN layer (from kan_curves.py), if it has been run

Data labels: "multi-f25" = 25% of the training photos, "base-rand" = photo-level random splits
(leakage check), "bbhl_cross" = Bacterial Blight vs Healthy trained on one source, tested on the other.
"test_acc_if_epoch_chosen_on_test" = what the old protocol would report (best epoch picked on the
test fold) - only for showing how much that inflates the numbers, never as a result.

Run any time (also while runs are still going):  python summarize.py
"""
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import RUNS_DIR, OUT_ROOT

SETTING_COLS = ["phase", "data", "preproc", "head", "hidden", "grid", "order", "dropout", "norm",
                "kan_base", "backbone", "unfreeze", "seed", "select", "epochs", "split", "train_frac", "oversample"]
EQUIV_MARGIN = {"test_acc": 2.0, "test_bal_acc": 2.0, "test_f1": 0.02}


def load_runs():
    runs, folds, kans, knock = [], [], [], []
    if not RUNS_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for d in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        cfg_p, folds_p = d / "config.json", d / "folds.csv"
        if not cfg_p.exists() or not folds_p.exists() or folds_p.stat().st_size < 5:
            continue
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        if cfg.get("smoke") in (True, "True"):
            continue
        f = pd.read_csv(folds_p)
        ep = d / "epochs.csv"
        if ep.exists() and ep.stat().st_size > 5:
            try:
                e = pd.read_csv(ep, usecols=["fold", "test_acc"])
                f["test_acc_if_epoch_chosen_on_test"] = f.fold.map(e.groupby("fold").test_acc.max())
            except (ValueError, pd.errors.EmptyDataError):
                pass
        settings = {k: cfg.get(k) for k in SETTING_COLS}
        settings["kan_base"] = settings["kan_base"] or "on"          # runs made before this option = on
        settings["split"] = settings["split"] or "group"
        settings["train_frac"] = float(settings["train_frac"] or 1.0)
        settings["data_raw"] = settings["data"]
        if settings["split"] == "random":
            settings["data"] += "-rand"
        if settings["train_frac"] < 1:
            settings["data"] += f"-f{round(settings['train_frac'] * 100)}"
        if settings["head"] == "linear":
            settings["hidden"] = "-"
        settings["run_name"] = d.name
        for k, v in settings.items():
            f[k] = v if not isinstance(v, list) else str(v)
        folds.append(f)
        info = {}
        mi = d / "run_log.xlsx"
        try:
            m = pd.read_excel(mi, sheet_name="model_info")
            info = dict(zip(m["item"], m["value"]))
        except Exception:
            pass
        row = dict(settings, status=(d / "STATUS.txt").read_text(encoding="utf-8").split("(")[0].strip()
                   if (d / "STATUS.txt").exists() else "", n_folds=len(f),
                   head_params=info.get("head_params"), head_trainable_params=info.get("head_trainable_params"),
                   total_params=info.get("total_params"),
                   macs_per_image=info.get("macs_per_image"))
        for c in ["test_acc", "test_f1", "test_bal_acc", "val_acc", "val_f1", "val_loss", "best_epoch",
                  "test_acc_last_epoch", "test_acc_mean_last5", "test_acc_if_epoch_chosen_on_test", "fold_minutes"]:
            if c in f:
                row[c + "_mean"] = f[c].mean()
                row[c + "_std"] = f[c].std(ddof=1) if len(f) > 1 else 0.0
        runs.append(row)
        kc = d / "kan_check.csv"
        if kc.exists() and kc.stat().st_size > 5:
            k = pd.read_csv(kc)
            k.insert(0, "run_name", d.name)
            kans.append(k)
        ko = d / "kan_knockout.csv"
        if ko.exists() and ko.stat().st_size > 5:
            k = pd.read_csv(ko)
            k.insert(0, "run_name", d.name)
            knock.append(k)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return pd.DataFrame(runs), cat(folds), cat(kans), cat(knock)


def rank(runs, phase):
    r = runs[runs.phase == phase].copy()
    if r.empty:
        return r
    cols = ["run_name", "data", "preproc", "backbone", "head", "hidden", "grid", "order", "dropout",
            "n_folds", "val_f1_mean", "val_f1_std", "val_acc_mean", "test_acc_mean", "test_f1_mean"]
    r = r.sort_values(["head", "val_f1_mean"], ascending=[True, False])
    r["rank_within_head (by val F1)"] = r.groupby("head").cumcount() + 1
    return r[[c for c in cols if c in r] + ["rank_within_head (by val F1)"]]


def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for k-fold x r repeats."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return np.nan, np.nan, (np.nan, np.nan)
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / n + n_test / n_train) * var)
    if se == 0:
        return np.nan, np.nan, (d.mean(), d.mean())
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    h = stats.t.ppf(0.975, df=n - 1) * se
    return t, p, (d.mean() - h, d.mean() + h)


def tost(d, n_train, n_test, margin):
    """Equivalence test with the same Nadeau-Bengio correction: the two heads count as equal within
    +-margin if both one-sided tests reject (p < 0.05), i.e. the 90% CI lies inside (-margin, +margin).
    Also returns the smallest margin that would pass."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return np.nan, (np.nan, np.nan), np.nan
    se = np.sqrt((1.0 / n + n_test / n_train) * d.var(ddof=1))
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0, (d.mean(), d.mean()), abs(d.mean())
    p_low = stats.t.sf((d.mean() + margin) / se, df=n - 1)       # H0: diff <= -margin
    p_high = stats.t.cdf((d.mean() - margin) / se, df=n - 1)     # H0: diff >= +margin
    h = stats.t.ppf(0.95, df=n - 1) * se
    return max(p_low, p_high), (d.mean() - h, d.mean() + h), abs(d.mean()) + h


def add_config_label(f):
    f = f.copy()
    nb = np.where(f["kan_base"].astype(str) == "off", "_nobase", "")
    kan = (f["head"] + "_h" + f["hidden"].astype(str) + "_g" + f["grid"].astype(str) + "_o" +
           f["order"].astype(str) + nb + "_d" + f["dropout"].astype(str) + "_" + f["norm"].astype(str))
    mlp = f["head"] + "_h" + f["hidden"].astype(str) + "_d" + f["dropout"].astype(str) + "_" + f["norm"].astype(str)
    lin = f["head"] + "_d" + f["dropout"].astype(str) + "_" + f["norm"].astype(str)
    f["config"] = np.where(f["head"] == "kan", kan, np.where(f["head"] == "mlp", mlp, lin))
    return f


def paired(A, B, extra):
    """A, B: fold rows indexed by (seed, fold). Returns one row per metric."""
    common = A.index.intersection(B.index)
    if len(common) < 2:
        return []
    rows = []
    for metric in ("test_acc", "test_f1", "test_bal_acc"):
        if metric not in A or metric not in B:
            continue
        d = (A.loc[common, metric] - B.loc[common, metric]).to_numpy(dtype=float)
        n_test = A.loc[common, "n_test"].mean()
        n_train = (A.loc[common, "n_train_real"] + A.loc[common, "n_val"]).mean()
        t, p, ci = corrected_ttest(d, n_train, n_test)
        margin = EQUIV_MARGIN[metric]
        p_eq, ci90, smallest = tost(d, n_train, n_test, margin)
        rows.append(dict(extra, metric=metric, n_pairs=len(d), seeds=len(set(s for s, _ in common)),
                         mean_A=A.loc[common, metric].mean(), mean_B=B.loc[common, metric].mean(),
                         mean_diff_A_minus_B=d.mean(), A_better_in=int((d > 0).sum()), B_better_in=int((d < 0).sum()),
                         ties=int((d == 0).sum()), ci95_low=ci[0], ci95_high=ci[1], corrected_t=t, corrected_p=p,
                         equivalence_margin=margin, equivalence_p=p_eq, equivalent=bool(p_eq < 0.05),
                         ci90_low=ci90[0], ci90_high=ci90[1], smallest_equivalent_margin=smallest))
    return rows


def comparisons(folds):
    """Compare every pair of heads that were run on the same data/backbone/unfreeze/phase,
    paired by (seed, fold)."""
    if folds.empty:
        return pd.DataFrame()
    f = add_config_label(folds)
    rows = []
    for key, g in f.groupby(["phase", "data", "preproc", "backbone", "unfreeze"]):
        configs = sorted(g.config.unique())
        for a, b in itertools.combinations(configs, 2):
            A = g[g.config == a].set_index(["seed", "fold"])
            B = g[g.config == b].set_index(["seed", "fold"])
            rows += paired(A, B, dict(phase=key[0], data=key[1], preproc=key[2], backbone=key[3], unfreeze=key[4],
                                      A=a, B=b))
    return pd.DataFrame(rows)


def other_comparisons(folds):
    """Same head, one other thing changed (the test images stay the same, so pairs are valid):
       multi_syn vs multi (synthetic data), layer4 vs frozen (fine-tuning)."""
    if folds.empty:
        return pd.DataFrame()
    f = add_config_label(folds)
    rows = []
    for factor, a, b in (("data", "multi_syn", "multi"), ("unfreeze", "layer4", "none")):
        others = [c for c in ["phase", "data", "preproc", "backbone", "unfreeze", "config"] if c != factor]
        for key, g in f.groupby(others):
            A = g[g[factor] == a].set_index(["seed", "fold"])
            B = g[g[factor] == b].set_index(["seed", "fold"])
            rows += paired(A, B, dict(zip(others, key), changed=factor, A=a, B=b))
    return pd.DataFrame(rows)


def by_config(folds, runs):
    """Every setting averaged over all its seeds and folds (one row = one line of a paper table)."""
    if folds.empty:
        return pd.DataFrame()
    f = add_config_label(folds)
    keys = ["phase", "data", "preproc", "backbone", "unfreeze", "head", "config"]
    agg = {}
    for c in ["test_acc", "test_f1", "test_bal_acc", "val_f1", "best_epoch", "test_acc_if_epoch_chosen_on_test",
              "fold_minutes"]:
        if c in f:
            agg[c + "_mean"] = (c, "mean")
            agg[c + "_std"] = (c, "std")
    out = f.groupby(keys).agg(n_seeds=("seed", "nunique"), n_fold_runs=("fold", "count"), **agg).reset_index()
    info = runs.drop_duplicates("run_name").copy()
    info = add_config_label(info.assign(fold=0))
    info = info.groupby(keys)[["head_params", "head_trainable_params", "total_params", "macs_per_image"]].first().reset_index()
    out = out.merge(info, on=keys, how="left")
    return out.sort_values(keys).reset_index(drop=True)


def knockout_summary(knock):
    if knock.empty:
        return knock
    return (knock.groupby(["run_name", "variant"], sort=False)
            .agg(test_acc_mean=("test_acc", "mean"), change_in_acc_mean=("change_in_acc", "mean"),
                 change_in_acc_min=("change_in_acc", "min"), change_in_acc_max=("change_in_acc", "max"),
                 test_f1_mean=("test_f1", "mean"), change_in_f1_mean=("change_in_f1", "mean"),
                 folds=("fold", "count")).reset_index())


def main():
    runs, folds, kans, knock = load_runs()
    if runs.empty:
        print("No finished runs found in", RUNS_DIR)
        return
    out = OUT_ROOT / "all_runs.xlsx"
    tmp = OUT_ROOT / "all_runs.tmp.xlsx"
    comp = comparisons(folds)
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        runs.to_excel(xw, sheet_name="runs", index=False)
        folds.to_excel(xw, sheet_name="folds", index=False)
        rank(runs, "P1").to_excel(xw, sheet_name="phase1_choice", index=False)
        rank(runs, "P2").to_excel(xw, sheet_name="phase2_choice", index=False)
        by_config(folds, runs).to_excel(xw, sheet_name="by_config", index=False)
        comp.to_excel(xw, sheet_name="comparisons", index=False)
        other_comparisons(folds).to_excel(xw, sheet_name="other_comparisons", index=False)
        kans.to_excel(xw, sheet_name="kan_check", index=False)
        knockout_summary(knock).to_excel(xw, sheet_name="kan_knockout", index=False)
        for sheet, path in (("speed", OUT_ROOT / "speed" / "speed.csv"),
                            ("kan_curves", OUT_ROOT / "figures" / "kan_curves_summary.csv")):
            if path.exists():
                pd.read_csv(path).to_excel(xw, sheet_name=sheet, index=False)
    tmp.replace(out)
    print(f"{len(runs)} runs, {len(folds)} fold rows -> {out}")
    show = [c for c in ["run_name", "n_folds", "val_f1_mean", "test_acc_mean", "test_acc_std", "test_f1_mean"] if c in runs]
    print(runs[show].sort_values("run_name").to_string(index=False))
    bc_all = by_config(folds, runs)
    for ph in ("P3", "P4"):
        bc = bc_all[bc_all.phase == ph] if not bc_all.empty else bc_all
        if not bc.empty:
            print(f"\nPhase {ph[1:]} so far (mean over all seeds x folds):")
            cols = ["data", "backbone", "unfreeze", "config", "n_seeds", "n_fold_runs", "test_acc_mean",
                    "test_acc_std", "test_f1_mean"]
            print(bc[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
