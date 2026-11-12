"""
make_syn2_split.py  -  split file for the synthetic set (dataset multi_syn2). The main splits are NOT changed.

The synthetic set is made from text prompts only, and it is used like this:

  * the real photos keep EXACTLY the roles they have in splits_multi.csv (same train/val/test in every fold);
  * the synthetic images come in 5 batches (a whole Gemini image, i.e. all its tiles, is always in one batch);
  * batch k is 'unused' in fold k and 'train' in the other folds, so every fold trains on 4 of the 5 batches.
    This keeps the synthetic images under 50% of each class's TRAINING images in every fold.
    (If samples are ever used, the images made from fold k's test photos go in batch k, so a fold never
    trains on images made from its own test photos.)
  * synthetic images are never validation or test images.

Reads   outputs/splits/splits_multi.csv
        outputs/metadata/synthetic_v2.csv      (written by process_synthetic.py finalize)
Writes  outputs/splits/splits_multi_syn2.csv
        outputs/splits/syn2_split_log.txt, outputs/splits/syn2_split.xlsx (counts, shares and checks)

Run:    python make_syn2_split.py            (refuses to overwrite without --force)
"""
import argparse
import datetime as dt

import pandas as pd

from common import SPLIT_DIR, META_DIR, N_OUTER_FOLDS, text_md5, Tee

COLS = ["uid", "cls", "source", "group_id", "is_synthetic", "file"]
MAX_SHARE = 0.5          # synthetic images must stay BELOW this share of a class's training images


def fingerprint(sp):
    fold_cols = [c for c in sp.columns if c.startswith("fold")]
    return text_md5(sp.sort_values("uid")[["uid"] + fold_cols].to_csv(index=False, lineterminator="\n"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits_multi_syn2.csv")
    args = ap.parse_args()
    out_csv = SPLIT_DIR / "splits_multi_syn2.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"{out_csv.name} already exists. Use --force only if no multi_syn2 run has started yet.")
    log = Tee(SPLIT_DIR / "syn2_split_log.txt")
    log(f"=== make_syn2_split.py  {dt.datetime.now():%Y-%m-%d %H:%M:%S}")

    multi = pd.read_csv(SPLIT_DIR / "splits_multi.csv", keep_default_na=False)
    multi["is_synthetic"] = multi.is_synthetic.astype(str).str.lower().isin(["true", "1"])
    fold_cols = [f"fold{k}" for k in range(N_OUTER_FOLDS)]
    real = multi[~multi.is_synthetic][COLS + fold_cols].copy()
    real["syn_batch"] = ""

    syn = pd.read_csv(META_DIR / "synthetic_v2.csv", keep_default_na=False)
    syn["batch"] = syn.batch.astype(int)
    s = pd.DataFrame(dict(uid=syn.uid, cls=syn.cls, source="GEMINI", group_id=syn.group_id, is_synthetic=True,
                          file=syn.file, syn_batch=syn.batch))
    for k in range(N_OUTER_FOLDS):
        s[f"fold{k}"] = ["unused" if b == k else "train" for b in syn.batch]
    sp = pd.concat([real, s[real.columns]], ignore_index=True)

    # ---------------- checks
    rows = []

    def add(check, ok, detail=""):
        rows.append(dict(check=check, passed=bool(ok), detail=detail))
        log(f"  [{'OK ' if ok else 'FAIL'}] {check} {detail}")

    m_real = multi[~multi.is_synthetic].set_index("uid")[fold_cols].sort_index()
    n_real = sp[~sp.is_synthetic].set_index("uid")[fold_cols].sort_index()
    add("real photos and their roles are identical to splits_multi.csv", m_real.equals(n_real),
        f"({len(n_real)} real photos)")
    add("synthetic uids are new", not set(s.uid) & set(real.uid))
    add("synthetic uids are unique", s.uid.is_unique)
    add("every Gemini image (group) is in one batch only", (syn.groupby("group_id").batch.nunique() == 1).all())
    add("batches are 0-4", set(syn.batch) <= set(range(N_OUTER_FOLDS)), str(sorted(set(syn.batch))))
    roles = s[fold_cols].stack()
    add("synthetic images are never val or test", set(roles) <= {"train", "unused"})
    add("batch k is unused in fold k and train elsewhere",
        all(((s[f"fold{k}"] == "unused") == (s.syn_batch == k)).all() for k in range(N_OUTER_FOLDS)))

    count_rows = []
    for k in range(N_OUTER_FOLDS):
        for cls in sorted(sp.cls.unique()):
            r = sp[(sp.cls == cls) & ~sp.is_synthetic]
            n_syn = int(((s.cls == cls) & (s[f"fold{k}"] == "train")).sum())
            n_tr = int((r[f"fold{k}"] == "train").sum())
            count_rows.append(dict(fold=k, cls=cls, train_real=n_tr, val=int((r[f"fold{k}"] == "val").sum()),
                                   test=int((r[f"fold{k}"] == "test").sum()), train_synthetic=n_syn,
                                   synthetic_share_of_training=round(n_syn / (n_syn + n_tr), 4) if n_syn else 0.0))
    counts = pd.DataFrame(count_rows)
    worst = counts.synthetic_share_of_training.max()
    add(f"synthetic images below {MAX_SHARE:.0%} of each class's training images in every fold", worst < MAX_SHARE,
        f"(highest share {worst:.1%})")
    batches = syn.groupby(["cls", "batch"]).size().unstack(fill_value=0)
    log("\nSynthetic images per class and batch:\n" + batches.to_string())
    log("\nTraining images per fold (synthetic classes only):\n" +
        counts[counts.train_synthetic > 0].to_string(index=False))

    checks = pd.DataFrame(rows)
    if not checks.passed.all():
        raise SystemExit("Some checks failed - nothing was written. See the log above.")
    sp.to_csv(out_csv, index=False, encoding="utf-8")
    fp = fingerprint(sp)
    with pd.ExcelWriter(SPLIT_DIR / "syn2_split.xlsx", engine="openpyxl") as xw:
        checks.to_excel(xw, sheet_name="checks", index=False)
        counts.to_excel(xw, sheet_name="counts", index=False)
        batches.reset_index().to_excel(xw, sheet_name="batches", index=False)
        pd.DataFrame([dict(file=out_csv.name, rows=len(sp), synthetic=len(s), fingerprint=fp)]).to_excel(
            xw, sheet_name="info", index=False)
    log(f"\nWrote {out_csv.name}: {len(sp)} rows ({len(s)} synthetic), fingerprint {fp}. All {len(checks)} checks passed.")


if __name__ == "__main__":
    main()
