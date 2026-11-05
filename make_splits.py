"""
make_splits.py  -  STEP 2 of the journal re-run

Reads outputs/metadata/metadata.csv (from build_metadata.py) and writes the
FIXED splits that every model and every seed will use:

  outputs/splits/splits_multi.csv   SesameMulti (4 classes) + synthetic rows (train only)
  outputs/splits/splits_base.csv    SesameBase  (3 classes)
  outputs/splits/splits.xlsx        both tables + image counts + checks
  outputs/splits/split_fingerprints.txt

How the split works (same for both datasets):
  * Outer: 5 folds. Each fold is the TEST set once (20%).
    Folds keep groups together (photos of the same leaf never get split up)
    and are balanced by class AND source (ERFD / MSLD / RSFD).
    The fold assignment is our own small greedy algorithm (numpy only), so the
    splits come out the same on every computer and library version.
  * Inner: inside each fold's training part, 1/5 becomes VALIDATION (16% of all),
    the rest is TRAIN (64% of all). Groups are kept together here too.
  * Synthetic images are 'train' in every fold. They are never validation or test.

Each table has one column per fold (fold0 ... fold4) with the role of the image:
'train', 'val' or 'test'.

The splits must never change once training starts, so this script refuses
to overwrite existing split files unless you pass --force.

Run:  python make_splits.py
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

from common import (META_DIR, SPLIT_DIR, CLASSES, SPLIT_SEED, N_OUTER_FOLDS, N_INNER_FOLDS,
                    text_md5, environment_info, Tee)


def balanced_group_folds(df, k, seed):
    """Put every GROUP into one of k folds so that each fold gets about 1/k of every
    class+source combination (stratum) and about 1/k of all images.

    Greedy: biggest groups first (random order among equal sizes), each one goes to the fold
    where it adds the least imbalance. Only uses numpy's legacy RandomState, so the result is
    the same on every computer and every library version.
    Returns a numpy array with the fold number of every row of df."""
    rs = np.random.RandomState(seed)
    strata = sorted(df.stratum.unique())
    s_index = {s: i for i, s in enumerate(strata)}
    groups = sorted(df.group_id.unique())
    g_index = {g: i for i, g in enumerate(groups)}
    vec = np.zeros((len(groups), len(strata)))
    for g, s in zip(df.group_id, df.stratum):
        vec[g_index[g], s_index[s]] += 1
    size = vec.sum(1)
    order = rs.permutation(len(groups))
    order = order[np.argsort(-size[order], kind="stable")]

    target = vec.sum(0) / k                       # wanted count per stratum and fold
    total_target = vec.sum() / k
    counts = np.zeros((k, len(strata)))
    fold_of_group = np.zeros(len(groups), dtype=int)
    for gi in order:
        best_f, best_cost = 0, None
        for f in range(k):
            new = counts.copy()
            new[f] += vec[gi]
            cost = (((new - target) ** 2) / np.maximum(target, 1)).sum() + \
                   (((new.sum(1) - total_target) ** 2) / total_target).sum()
            if best_cost is None or cost < best_cost - 1e-12:
                best_f, best_cost = f, cost
        counts[best_f] += vec[gi]
        fold_of_group[gi] = best_f
    return np.array([fold_of_group[g_index[g]] for g in df.group_id])


def pick_inner_split(train_part, seed):
    """Group-aware inner split of one training part into N_INNER_FOLDS pieces; the most
    balanced piece becomes the validation set."""
    folds = balanced_group_folds(train_part, N_INNER_FOLDS, seed)
    target = train_part.cls.value_counts() / N_INNER_FOLDS
    best, best_score = None, None
    for f in range(N_INNER_FOLDS):
        val_idx = np.where(folds == f)[0]
        counts = train_part.iloc[val_idx].cls.value_counts().reindex(target.index, fill_value=0)
        score = float(((counts - target).abs() / target).max())
        if best_score is None or score < best_score:
            best, best_score = val_idx, score
    return best, best_score


def make_split(meta, dataset, log):
    if dataset == "multi":
        real = meta[meta.in_multi & ~meta.is_synthetic].copy()
        syn = meta[meta.in_multi & meta.is_synthetic].copy()
        file_col = "file"
    else:
        real = meta[meta.in_base & ~meta.is_synthetic].copy()
        syn = meta.iloc[0:0].copy()
        file_col = "file_base"
    real = real.sort_values("uid").reset_index(drop=True)      # order-independent
    real["file_used"] = real[file_col].where(real[file_col].astype(str) != "", real["file"])
    real["stratum"] = real.cls + "|" + real.source
    syn["file_used"] = syn["file"]

    outer_fold = balanced_group_folds(real, N_OUTER_FOLDS, SPLIT_SEED)
    roles = pd.DataFrame(index=real.index)
    inner_scores = []
    for k in range(N_OUTER_FOLDS):
        role = np.array(["train"] * len(real), dtype=object)
        test_idx = np.where(outer_fold == k)[0]
        train_idx = np.where(outer_fold != k)[0]
        role[test_idx] = "test"
        train_part = real.iloc[train_idx]
        val_local, score = pick_inner_split(train_part, SPLIT_SEED + 100 + k)
        role[train_idx[val_local]] = "val"
        roles[f"fold{k}"] = role
        inner_scores.append(score)

    out = pd.concat([real[["uid", "cls", "source", "group_id", "is_synthetic", "file_used"]], roles], axis=1)
    if len(syn):
        s = syn.sort_values("uid")[["uid", "cls", "source", "group_id", "is_synthetic", "file_used"]].copy()
        for k in range(N_OUTER_FOLDS):
            s[f"fold{k}"] = "train"
        out = pd.concat([out, s], ignore_index=True)
    out = out.rename(columns={"file_used": "file"})
    log(f"[{dataset}] real images: {len(real)}  synthetic: {len(syn)}  groups: {real.group_id.nunique()}")
    return out, inner_scores


def check_split(sp, dataset, log):
    fold_cols = [f"fold{k}" for k in range(N_OUTER_FOLDS)]
    real = sp[~sp.is_synthetic]
    rows = []

    def add(name, ok, detail=""):
        rows.append(dict(dataset=dataset, check=name, passed=bool(ok), detail=str(detail)))

    n_test = (real[fold_cols] == "test").sum(axis=1)
    add("every real image is TEST in exactly one fold", (n_test == 1).all(),
        f"min={n_test.min()} max={n_test.max()}")
    for c in fold_cols:
        per_group = real.groupby("group_id")[c].nunique()
        add(f"{c}: no group is split across train/val/test", (per_group == 1).all(),
            f"{int((per_group > 1).sum())} groups split")
        for role in ("train", "val", "test"):
            present = set(real.loc[real[c] == role, "cls"])
            expected = set(real.cls)
            add(f"{c}: every class present in {role}", present == expected,
                f"missing: {sorted(expected - present)}")
    syn = sp[sp.is_synthetic]
    if len(syn):
        add("synthetic images are never val/test", (syn[fold_cols] == "train").all().all())
    add("uids are unique", sp.uid.is_unique)

    checks = pd.DataFrame(rows)
    bad = checks[~checks.passed]
    log(f"[{dataset}] checks passed: {int(checks.passed.sum())}/{len(checks)}")
    for r in bad.itertuples():
        log(f"   FAILED: {r.check} ({r.detail})")
    return checks


def count_tables(sp, dataset):
    fold_cols = [f"fold{k}" for k in range(N_OUTER_FOLDS)]
    long = sp.melt(id_vars=["uid", "cls", "source", "is_synthetic"], value_vars=fold_cols,
                   var_name="fold", value_name="role")
    long["kind"] = np.where(long.is_synthetic, "synthetic", "real")
    by_class = long.pivot_table(index=["fold", "role"], columns=["kind", "cls"], values="uid",
                                aggfunc="count", fill_value=0)
    by_class["total"] = by_class.sum(axis=1)
    by_source = long[~long.is_synthetic].pivot_table(index=["fold", "role"], columns=["cls", "source"],
                                                     values="uid", aggfunc="count", fill_value=0)
    # role shares
    share = long[~long.is_synthetic].groupby(["fold", "role"]).uid.count().unstack()
    share = (share.div(share.sum(axis=1), axis=0) * 100).round(1)
    return by_class, by_source, share


def fingerprint(sp):
    fold_cols = [f"fold{k}" for k in range(N_OUTER_FOLDS)]
    text = sp.sort_values("uid")[["uid"] + fold_cols].to_csv(index=False, lineterminator="\n")  # same on Windows and Linux
    return text_md5(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    log = Tee(SPLIT_DIR / "make_splits_log.txt")
    log("=" * 70)
    log(f"make_splits.py  started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")

    targets = [SPLIT_DIR / "splits_multi.csv", SPLIT_DIR / "splits_base.csv"]
    if any(t.exists() for t in targets) and not args.force:
        raise SystemExit("Split files already exist. They must stay fixed once training starts.\n"
                         "Only if NO training has been run yet, re-run with:  python make_splits.py --force")

    meta_path = META_DIR / "metadata.csv"
    if not meta_path.exists():
        raise SystemExit("metadata.csv not found - run build_metadata.py first.")
    meta = pd.read_csv(meta_path, keep_default_na=False)
    for col in ("is_synthetic", "in_multi", "in_base", "excluded"):
        if col not in meta:
            meta[col] = False
        meta[col] = meta[col].astype(str).str.lower().isin(["true", "1"])
    if meta.excluded.any():
        log(f"Leaving out {int(meta.excluded.sum())} excluded images (exclude.csv):")
        for r in meta[meta.excluded].itertuples():
            log(f"   {r.uid}  - {r.exclude_reason}")
        meta = meta[~meta.excluded].reset_index(drop=True)
    log(f"metadata: {len(meta)} rows, fingerprint of metadata.csv = {text_md5(meta_path.read_text(encoding='utf-8'))}")

    results = {}
    for dataset in ("multi", "base"):
        sp, inner_scores = make_split(meta, dataset, log)
        checks = check_split(sp, dataset, log)
        by_class, by_source, share = count_tables(sp, dataset)
        fp = fingerprint(sp)
        log(f"[{dataset}] split fingerprint: {fp}")
        log(f"[{dataset}] role shares (% of real images) per fold:\n{share.to_string()}")
        sp.to_csv(SPLIT_DIR / f"splits_{dataset}.csv", index=False, encoding="utf-8")
        results[dataset] = dict(sp=sp, checks=checks, by_class=by_class, by_source=by_source,
                                share=share, fp=fp, inner=inner_scores)

    with open(SPLIT_DIR / "split_fingerprints.txt", "w", encoding="utf-8") as f:
        for d, r in results.items():
            f.write(f"{d}\t{r['fp']}\n")

    info = pd.DataFrame(
        [dict(item="split seed", value=SPLIT_SEED), dict(item="outer folds", value=N_OUTER_FOLDS),
         dict(item="inner folds (validation = 1/n of training part)", value=N_INNER_FOLDS),
         dict(item="stratified by", value="class + source"), dict(item="grouped by", value="group_id"),
         dict(item="created", value=dt.datetime.now().isoformat(timespec="seconds"))] +
        [dict(item=f"fingerprint {d}", value=r["fp"]) for d, r in results.items()] +
        [dict(item=f"{d} inner-split worst class deviation fold{k}", value=round(s, 3))
         for d, r in results.items() for k, s in enumerate(r["inner"])] +
        [dict(item="env." + k, value=str(v)) for k, v in environment_info().items()])

    with pd.ExcelWriter(SPLIT_DIR / "splits.xlsx", engine="openpyxl") as xw:
        info.to_excel(xw, sheet_name="info", index=False)
        pd.concat([r["checks"] for r in results.values()]).to_excel(xw, sheet_name="checks", index=False)
        for d, r in results.items():
            r["sp"].to_excel(xw, sheet_name=f"{d}_table", index=False)
            r["by_class"].to_excel(xw, sheet_name=f"{d}_counts_class")
            r["by_source"].to_excel(xw, sheet_name=f"{d}_counts_source")
            r["share"].to_excel(xw, sheet_name=f"{d}_role_shares")
    log(f"Saved: {SPLIT_DIR / 'splits.xlsx'}")
    if any(not r["checks"].passed.all() for r in results.values()):
        log("WARNING: some checks FAILED - look at the 'checks' sheet before training.")
    log("Done")


if __name__ == "__main__":
    main()
