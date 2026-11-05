"""
make_extra_splits.py  -  extra split files for phase 4. The main splits are NOT changed.

Reads outputs/splits/splits_multi.csv and splits_base.csv and writes, in outputs/splits/:

  splits_multi_random.csv, splits_base_random.csv
      5 folds with the same 64/16/20 sizes as the main splits, but every PHOTO is placed on
      its own and the near-duplicate groups are ignored (the way the GP split the data).
      Only used to measure how much near-duplicate leakage inflates the scores.

  splits_bbhl_cross.csv   (source check)
      Bacterial Blight vs Healthy, photos from ERFD and MSLD only (the two sources that have both classes).
        fold0: train + validation from ERFD, test = MSLD
        fold1: train + validation from MSLD, test = ERFD
      Validation = a group-aware 1/5 of the training source.
      Test photos whose near-duplicate group also has a photo in the training source are left out ('-').

  splits_bbhl_mixed.csv   (reference for the source check)
      The same photos with the normal 5-fold roles copied from splits_multi.csv.

  extra_splits_log.txt, extra_splits.xlsx (counts per fold and role, and checks)

Run once:  python make_extra_splits.py        (refuses to overwrite without --force)
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

from common import SPLIT_DIR, SPLIT_SEED, N_OUTER_FOLDS, text_md5, Tee
from make_splits import balanced_group_folds, pick_inner_split

BB, HL = "Bacteria Blight Infected", "Healthy Leaf"
BASE_COLS = ["uid", "cls", "source", "group_id", "is_synthetic", "file"]


def read_split(name):
    sp = pd.read_csv(SPLIT_DIR / name, keep_default_na=False)
    sp["is_synthetic"] = sp.is_synthetic.astype(str).str.lower().isin(["true", "1"])
    return sp


def fingerprint(sp):
    fold_cols = [c for c in sp.columns if c.startswith("fold")]
    return text_md5(sp.sort_values("uid")[["uid"] + fold_cols].to_csv(index=False, lineterminator="\n"))


def random_split(sp, log, name):
    """Photo-level split: each photo is its own 'group'."""
    real = sp[~sp.is_synthetic].sort_values("uid").reset_index(drop=True)
    work = real.copy()
    work["group_id"] = work.uid                     # ignore the near-duplicate groups on purpose
    work["stratum"] = work.cls + "|" + work.source
    outer = balanced_group_folds(work, N_OUTER_FOLDS, SPLIT_SEED)
    out = real[BASE_COLS].copy()
    for k in range(N_OUTER_FOLDS):
        role = np.array(["train"] * len(work), dtype=object)
        test_idx = np.where(outer == k)[0]
        train_idx = np.where(outer != k)[0]
        role[test_idx] = "test"
        val_local, _ = pick_inner_split(work.iloc[train_idx], SPLIT_SEED + 100 + k)
        role[train_idx[val_local]] = "val"
        out[f"fold{k}"] = role
    syn = sp[sp.is_synthetic].sort_values("uid")[BASE_COLS].copy()
    for k in range(N_OUTER_FOLDS):
        syn[f"fold{k}"] = "train"
    out = pd.concat([out, syn], ignore_index=True)

    # how many test photos have a near-duplicate (same group) in train or val of the same fold
    leaky = []
    for k in range(N_OUTER_FOLDS):
        c = f"fold{k}"
        r = out[~out.is_synthetic]
        seen = set(r.loc[r[c] != "test", "group_id"])
        t = r[r[c] == "test"]
        leaky.append(int(t.group_id.isin(seen).sum()))
    n_test = int((out[~out.is_synthetic]["fold0"] == "test").sum())
    log(f"[{name}] test photos with a near-duplicate in train/val, per fold: {leaky} "
        f"(of about {n_test} test photos per fold)")
    return out, leaky


def bbhl_rows(multi):
    r = multi[~multi.is_synthetic & multi.cls.isin([BB, HL]) & multi.source.isin(["ERFD", "MSLD"])]
    return r.sort_values("uid").reset_index(drop=True)


def bbhl_cross(multi, log):
    rows = bbhl_rows(multi)
    out = rows[BASE_COLS].copy()
    dropped = {}
    for k, (train_src, test_src) in enumerate([("ERFD", "MSLD"), ("MSLD", "ERFD")]):
        role = np.array(["-"] * len(rows), dtype=object)
        tr_idx = np.where(rows.source == train_src)[0]
        part = rows.iloc[tr_idx].copy()
        part["stratum"] = part.cls + "|" + part.source
        val_local, _ = pick_inner_split(part, SPLIT_SEED + 200 + k)
        role[tr_idx] = "train"
        role[tr_idx[val_local]] = "val"
        train_groups = set(part.group_id)
        te_idx = np.where(rows.source == test_src)[0]
        clash = rows.iloc[te_idx].group_id.isin(train_groups).to_numpy()
        role[te_idx[~clash]] = "test"
        dropped[f"fold{k}"] = int(clash.sum())
        out[f"fold{k}"] = role
        log(f"[bbhl_cross] fold{k}: train/val = {train_src}, test = {test_src}; "
            f"test photos left out because a near-duplicate is in {train_src}: {int(clash.sum())}")
    return out, dropped


def bbhl_mixed(multi):
    rows = bbhl_rows(multi)
    cols = BASE_COLS + [f"fold{k}" for k in range(N_OUTER_FOLDS)]
    return rows[cols].copy()


def checks(sp, name, all_test_once):
    fold_cols = [c for c in sp.columns if c.startswith("fold")]
    real = sp[~sp.is_synthetic]
    rows = []

    def add(check, ok, detail=""):
        rows.append(dict(split=name, check=check, passed=bool(ok), detail=str(detail)))

    if all_test_once:
        n_test = (real[fold_cols] == "test").sum(axis=1)
        add("every real photo is TEST in exactly one fold", (n_test == 1).all(), f"min={n_test.min()} max={n_test.max()}")
    for c in fold_cols:
        for role in ("train", "val", "test"):
            present = set(real.loc[real[c] == role, "cls"])
            add(f"{c}: every class present in {role}", present == set(real.cls), sorted(set(real.cls) - present))
        if "random" not in name:
            used = real[real[c].isin(["train", "val", "test"])]
            per_group = used.groupby("group_id")[c].nunique()
            add(f"{c}: no near-duplicate group split across train/val/test", (per_group == 1).all(),
                f"{int((per_group > 1).sum())} groups split")
        if name == "bbhl_cross":
            tr_src = set(real.loc[real[c].isin(["train", "val"]), "source"])
            te_src = set(real.loc[real[c] == "test", "source"])
            add(f"{c}: train and test come from different sources", not (tr_src & te_src), f"{tr_src} vs {te_src}")
    syn = sp[sp.is_synthetic]
    if len(syn):
        add("synthetic photos are never val/test", (syn[fold_cols] == "train").all().all())
    add("uids are unique", sp.uid.is_unique)
    return pd.DataFrame(rows)


def counts(sp, name):
    fold_cols = [c for c in sp.columns if c.startswith("fold")]
    long = sp[~sp.is_synthetic].melt(id_vars=["uid", "cls", "source"], value_vars=fold_cols,
                                     var_name="fold", value_name="role")
    t = long.pivot_table(index=["fold", "role"], columns=["cls", "source"], values="uid",
                         aggfunc="count", fill_value=0)
    t.columns = [f"{a} | {b}" for a, b in t.columns]
    t["total"] = t.sum(axis=1)
    t = t.reset_index()
    t.insert(0, "split", name)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    log = Tee(SPLIT_DIR / "extra_splits_log.txt")
    log("=" * 70)
    log(f"make_extra_splits.py  started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    targets = {n: SPLIT_DIR / f"splits_{n}.csv" for n in ("multi_random", "base_random", "bbhl_cross", "bbhl_mixed")}
    if any(p.exists() for p in targets.values()) and not args.force:
        raise SystemExit("Extra split files already exist. Use --force only if no phase 4 run has started.")

    multi, base = read_split("splits_multi.csv"), read_split("splits_base.csv")
    log(f"source files: splits_multi.csv fingerprint {fingerprint(multi)}, splits_base.csv fingerprint {fingerprint(base)}")

    results, info = {}, []
    results["multi_random"], leak_m = random_split(multi, log, "multi_random")
    results["base_random"], leak_b = random_split(base, log, "base_random")
    results["bbhl_cross"], dropped = bbhl_cross(multi, log)
    results["bbhl_mixed"] = bbhl_mixed(multi)

    all_checks, all_counts = [], []
    for name, sp in results.items():
        sp.to_csv(targets[name], index=False, encoding="utf-8")
        ch = checks(sp, name, all_test_once=(name != "bbhl_cross"))
        all_checks.append(ch)
        all_counts.append(counts(sp, name))
        fp = fingerprint(sp)
        log(f"[{name}] {len(sp)} rows, checks passed {int(ch.passed.sum())}/{len(ch)}, fingerprint {fp}")
        for r in ch[~ch.passed].itertuples():
            log(f"   FAILED: {r.check} ({r.detail})")
        info.append(dict(split=name, rows=len(sp), fingerprint=fp, checks_passed=f"{int(ch.passed.sum())}/{len(ch)}"))
    info += [dict(split="multi_random", rows="", fingerprint="", checks_passed=f"leaky test photos per fold: {leak_m}"),
             dict(split="base_random", rows="", fingerprint="", checks_passed=f"leaky test photos per fold: {leak_b}"),
             dict(split="bbhl_cross", rows="", fingerprint="", checks_passed=f"test photos left out: {dropped}")]
    with pd.ExcelWriter(SPLIT_DIR / "extra_splits.xlsx", engine="openpyxl") as xw:
        pd.DataFrame(info).to_excel(xw, sheet_name="info", index=False)
        pd.concat(all_checks).to_excel(xw, sheet_name="checks", index=False)
        pd.concat(all_counts).to_excel(xw, sheet_name="counts", index=False)
    log("Done")


if __name__ == "__main__":
    main()
