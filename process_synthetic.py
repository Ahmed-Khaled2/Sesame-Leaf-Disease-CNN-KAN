"""
process_synthetic.py  -  turn the Gemini images of the synthetic set into training tiles.

Folders (all under "Everything GP related/Sesame Datasets/Synthetic Data v2/"), for each class
("Phyllody Infected", "Powdery Mildew"):
    raw/<class>/          the images exactly as downloaded from the Gemini app (png, jpg or webp)
    tiles/<class>/        step 1: one file per tile, full size, Gemini logo covered
    rejected/<class>/     move any tile you do not want here by hand (after looking at review/)
    review/               step 1: contact sheets of the tiles, and the bottom-right corners (logo check)
    trial_a_224/<class> - Synthetic/   step 2: the final tiles at 224 x 224 (used for training)

STEP 1   python process_synthetic.py split
   * a raw image that is a 2 x 2 grid is cut into 4 tiles (found from the gutter or the seam between the
     panels), two panels side by side into 2, any other image is kept whole. Check the result in review/.
     A wrong guess can be fixed in "Synthetic Data v2/layout_overrides.csv" (columns cls,raw_file,layout with
     layout = grid (cut in 4 equal parts) or single) and running   python process_synthetic.py split --redo
   * the Gemini logo in the bottom-right corner is covered with the patch just above it
     (--always-cover-logo covers that corner on every image)
   * the thin white frame around each panel is trimmed off, so no tile has a white edge
   * tile names: syn2_<PI|PM>_<raw number>_<tl|tr|bl|br|l|r|t|b|full>.png. Raw numbers follow the order the files
     were saved in and never change, so a tile that you moved to rejected/ stays rejected.
   * outputs/metadata/synthetic_v2_raw.csv: every raw image, its layout and its tiles
   You can run step 1 again after adding more raw images: only the new ones are processed.

STEP 2   python process_synthetic.py finalize            (after removing the tiles you do not want)
   * drops any tile that is a near-copy of a real SesameMulti photo (ResNet50 feature similarity >= 0.97 in
     any orientation, the near-copy rule of build_metadata.py) and near-identical tiles (keeps the first)
   * keeps exactly --phyllody (default 75) and --powdery (default 111) tiles, picked at random with a fixed
     seed if there are more (it stops and tells you how many are missing if there are fewer)
   * puts whole Gemini images into 5 batches with about the same number of tiles (batch k = not used in fold k)
   * writes trial_a_224/, outputs/metadata/synthetic_v2.csv (+ _nearcopy.csv, _duplicates.csv,
     _features.npz for the UMAP figure)
   Then:  python make_syn2_split.py
"""
import argparse
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import GP_ROOT, DATASETS, META_DIR, SPLIT_DIR, CLASS_CODE, SYN_SUFFIX, make_uid, Tee

SYN2_ROOT = DATASETS / "Synthetic Data v2"
SYN_CLASSES = ["Phyllody Infected", "Powdery Mildew"]
RAW_EXT = (".png", ".jpg", ".jpeg", ".webp")
NEAR_COPY = 0.97            # same as --copy-cos in build_metadata.py
SEED = 2026
N_BATCHES = 5
RAW_TABLE = META_DIR / "synthetic_v2_raw.csv"
FINAL_TABLE = META_DIR / "synthetic_v2.csv"
RAW_COLS = ["cls", "raw_n", "raw_file", "raw_size", "layout", "logo_covered", "tiles", "added"]


def folder(kind, cls=""):
    return SYN2_ROOT / kind / cls if cls else SYN2_ROOT / kind


# ==========================================================================
# Step 1 helpers
# ==========================================================================
def cover_logo(img, force=False):
    """Cover a bright logo in the bottom-right corner with the patch just above it."""
    a = np.asarray(img)
    h, w, _ = a.shape
    b = max(8, int(round(0.09 * min(w, h))))
    corner = a[h - b:, w - b:].astype(np.float32) / 255
    above = a[h - 2 * b:h - b, w - b:].astype(np.float32) / 255

    def bright(x):
        mx, mn = x.max(2), x.min(2)
        return float(((mx > 0.85) & (mx - mn < 0.15)).mean())

    found = force or bright(corner) > 3 * bright(above) + 0.004
    if found:
        a = a.copy()
        a[h - b:, w - b:] = a[h - 2 * b:h - b, w - b:]
        img = Image.fromarray(a)
    return img, found


def find_boundary(g, z_min=25):
    """g: 2-D gray image (0-1). Returns (start, end) of the boundary between a left and a right panel
    (start = first gutter column, end = first column of the right panel), or None if there is none."""
    H, W = g.shape
    lo, hi = int(W * 0.42), int(W * 0.58)
    std = g.std(0)
    flat = np.where(std[lo:hi] < 0.015)[0] + lo            # a plain gutter between the panels
    if len(flat) >= 2:
        runs = np.split(flat, np.where(np.diff(flat) > 1)[0] + 1)
        run = min(runs, key=lambda r: abs(r.mean() - W / 2))
        if len(run) >= 2:
            return int(run[0]), int(run[-1]) + 1
    # a hard seam: along the whole height the image changes in one step. Measured as a robust z-score of the
    # mean column-to-column change; seams of 2 x 2 grids score far above 25, single photos stay below 10
    diff = np.abs(np.diff(g, axis=1)).mean(0)
    med = np.median(diff)
    mad = 1.4826 * np.median(np.abs(diff - med)) + 1e-6
    lo2, hi2 = int(W * 0.47), int(W * 0.53)
    z = (diff[lo2:hi2] - med) / mad
    x = int(z.argmax())
    if z[x] > z_min:
        return lo2 + x + 1, lo2 + x + 1
    return None


def trim_border(img, margin=2, cap=0.08):
    """Remove the plain white frame Gemini draws around each panel, plus a 2-pixel margin for its soft edge.
    Only pure white, even lines are cut (at most 8% per side), so a bright sky is left alone. Without this,
    every synthetic tile would carry a white edge that no real photo has."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255
    H, W = g.shape

    def run(lines, n_max):
        n = 0
        while n < n_max and lines[n].mean() > 0.93 and lines[n].std() < 0.05:
            n += 1
        return n

    t, b = run(g, int(H * cap)), run(g[::-1], int(H * cap))
    l, r = run(g.T, int(W * cap)), run(g.T[::-1], int(W * cap))
    return img.crop((l + margin, t + margin, W - r - margin, H - b - margin))


def split_grid(img):
    """2 x 2 grid -> tl, tr, bl, br. Two panels side by side (or one above the other) -> l, r (or t, b).
    A clear boundary on one axis makes a weaker one on the other axis count too. None = a single photo."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255
    v, h = find_boundary(g), find_boundary(g.T)
    if v is None and h is None:
        return None
    if v is None:
        v = find_boundary(g, z_min=8)
    if h is None:
        h = find_boundary(g.T, z_min=8)
    W, H = img.size
    if v is not None and h is not None:
        (v0, v1), (h0, h1) = v, h
        boxes = {"tl": (0, 0, v0, h0), "tr": (v1, 0, W, h0), "bl": (0, h1, v0, H), "br": (v1, h1, W, H)}
    elif v is not None:
        boxes = {"l": (0, 0, v[0], H), "r": (v[1], 0, W, H)}
    else:
        boxes = {"t": (0, 0, W, h[0]), "b": (0, h[1], W, H)}
    return {k: img.crop(b) for k, b in boxes.items()}


def contact_sheets(cls, log, per_sheet=30, cols=6, thumb=220):
    rev = folder("review")
    rev.mkdir(parents=True, exist_ok=True)
    code = CLASS_CODE[cls]
    tiles = sorted(folder("tiles", cls).glob("*.png"))
    n_sheets = (len(tiles) + per_sheet - 1) // per_sheet
    for old in rev.glob(f"{code}_tiles_*.jpg"):          # sheets left over from a longer run
        if int(old.stem.split("_")[-1]) > n_sheets:
            try:
                old.unlink()
            except OSError:
                pass
    for s in range(0, len(tiles), per_sheet):
        chunk = tiles[s:s + per_sheet]
        nrows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, nrows * (thumb + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for i, t in enumerate(chunk):
            im = Image.open(t).convert("RGB")
            im.thumbnail((thumb - 4, thumb - 4))
            x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
            sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
            d.text((x + 4, y + thumb + 2), t.stem, fill="black")
        sheet.save(rev / f"{code}_tiles_{s // per_sheet + 1:02d}.jpg", quality=88)
    log(f"  review sheets for {cls}: {len(tiles)} tiles -> {rev}")


def corner_sheet(cls, rows, cols=8, size=160):
    """Bottom-right corner of every raw image after the logo cover, enlarged (to check for logos)."""
    items = [r for r in rows if r["cls"] == cls]
    if not items:
        return
    nrows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, nrows * (size + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(items):
        last = r["tiles"].split()[-1]                        # br or full = the tile with the corner
        p = folder("tiles", cls) / last
        if not p.exists():
            p = folder("rejected", cls) / last
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        c = max(16, int(0.2 * min(w, h)))
        im = im.crop((w - c, h - c, w, h)).resize((size, size))
        x, y = (i % cols) * size, (i // cols) * (size + 16)
        sheet.paste(im, (x, y))
        d.text((x + 3, y + size + 1), f"{r['raw_n']:04d}", fill="black")
    sheet.save(folder("review") / f"{CLASS_CODE[cls]}_corners.jpg", quality=88)


def cmd_split(args, log):
    META_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RAW_TABLE, keep_default_na=False).to_dict("records") if RAW_TABLE.exists() else []
    for r in rows:
        r["raw_n"] = int(r["raw_n"])
    ov_path = SYN2_ROOT / "layout_overrides.csv"
    overrides = {}
    if ov_path.exists():
        ov = pd.read_csv(ov_path, keep_default_na=False)
        overrides = {(c, f): l.strip().lower() for c, f, l in zip(ov.cls, ov.raw_file, ov.layout)}
    for cls in SYN_CLASSES:
        raw_dir = folder("raw", cls)
        if not raw_dir.is_dir():
            log(f"  {raw_dir} not found - skipped")
            continue
        out_dir, rej_dir = folder("tiles", cls), folder("rejected", cls)
        out_dir.mkdir(parents=True, exist_ok=True)
        rej_dir.mkdir(parents=True, exist_ok=True)
        known = {r["raw_file"]: r for r in rows if r["cls"] == cls}
        files = sorted((p for p in raw_dir.iterdir() if p.suffix.lower() in RAW_EXT),
                       key=lambda p: (p.stat().st_mtime, p.name))
        next_n = max([r["raw_n"] for r in known.values()] or [0]) + 1
        for p in files:
            old = known.get(p.name)
            if old is not None and not args.redo:
                continue
            img = Image.open(p).convert("RGB")
            img, logo = cover_logo(img, force=args.always_cover_logo)
            want = overrides.get((cls, p.name), "auto")
            tiles = None if want == "single" else split_grid(img)
            if tiles is None and want == "grid":            # forced grid: plain halves
                W, H = img.size
                tiles = {"tl": img.crop((0, 0, W // 2, H // 2)), "tr": img.crop((W // 2, 0, W, H // 2)),
                         "bl": img.crop((0, H // 2, W // 2, H)), "br": img.crop((W // 2, H // 2, W, H))}
            layout = {4: "grid", 2: "pair"}.get(len(tiles), "single") if tiles else "single"
            tiles = tiles or {"full": img}
            n = old["raw_n"] if old is not None else next_n
            if old is None:
                next_n += 1
            names = []
            for pos, t in tiles.items():
                name = f"syn2_{CLASS_CODE[cls]}_{n:04d}_{pos}.png"
                if not (rej_dir / name).exists():
                    trim_border(t).save(out_dir / name)
                names.append(name)
            if old is not None and old["layout"] != layout:  # layout changed: remove tiles of the old layout
                for name in old["tiles"].split():
                    if name not in names and (out_dir / name).exists():
                        try:
                            (out_dir / name).unlink()
                        except OSError:
                            log(f"    could not remove the old tile {name} - delete it by hand")
            row = dict(cls=cls, raw_n=n, raw_file=p.name, raw_size=f"{img.size[0]}x{img.size[1]}", layout=layout,
                       logo_covered=logo, tiles=" ".join(names), added=dt.datetime.now().isoformat(timespec="seconds"))
            if old is not None:
                old.update(row)
            else:
                rows.append(row)
            log(f"  {cls}: {p.name} -> #{n:04d} {layout}, {len(names)} tile(s){', logo covered' if logo else ''}")
    pd.DataFrame(rows, columns=RAW_COLS).sort_values(["cls", "raw_n"]).to_csv(RAW_TABLE, index=False)
    for cls in SYN_CLASSES:
        if folder("tiles", cls).is_dir():
            contact_sheets(cls, log)
            corner_sheet(cls, rows)
    for cls in SYN_CLASSES:
        n_raw = sum(r["cls"] == cls for r in rows)
        n_tiles = len(list(folder("tiles", cls).glob("*.png"))) if folder("tiles", cls).is_dir() else 0
        n_rej = len(list(folder("rejected", cls).glob("*.png"))) if folder("rejected", cls).is_dir() else 0
        log(f"{cls}: {n_raw} raw images, {n_tiles} tiles kept, {n_rej} rejected")


# ==========================================================================
# Step 2
# ==========================================================================
def best_cos(A, B):
    """A: (n, 8, d), B: (m, 8, d), unit vectors -> (n, m) best cosine over the 8 orientations."""
    out = np.full((len(A), len(B)), -1.0, dtype=np.float32)
    for t in range(8):
        out = np.maximum(out, A[:, 0, :] @ B[:, t, :].T)
        out = np.maximum(out, A[:, t, :] @ B[:, 0, :].T)
    return out


def balanced_batches(sizes, rng):
    """sizes: {group: n tiles}. Whole groups go to the batch with the fewest tiles so far."""
    keys = list(sizes)
    rnd = dict(zip(keys, rng.permutation(len(keys))))
    order = sorted(keys, key=lambda g: (-sizes[g], rnd[g]))
    load, out = [0] * N_BATCHES, {}
    for g in order:
        b = int(np.argmin(load))
        out[g] = b
        load[b] += sizes[g]
    return out


def cmd_finalize(args, log):
    from build_metadata import extract_features
    targets = {"Phyllody Infected": args.phyllody, "Powdery Mildew": args.powdery}
    raw = pd.read_csv(RAW_TABLE, keep_default_na=False)
    raw_file = {(c, int(n)): f for c, n, f in zip(raw.cls, raw.raw_n, raw.raw_file)}
    cand = []
    for cls in SYN_CLASSES:
        for p in sorted(folder("tiles", cls).glob("*.png")):
            parts = p.stem.split("_")                       # syn2, PI, 0007, tl
            cand.append(dict(cls=cls, name=p.stem, raw_n=int(parts[2]), pos=parts[3],
                             tile_file=str(p.relative_to(GP_ROOT))))
    cand = pd.DataFrame(cand)
    if cand.empty:
        raise SystemExit("No tiles found - run   python process_synthetic.py split   first.")
    feats, _, md5s, sizes = extract_features(cand.tile_file.tolist(), SimpleNamespace(device=args.device, batch=32),
                                             log, "synthetic tiles")
    feats = feats.astype(np.float32)
    cand["md5"], cand["tile_size"] = md5s, sizes

    # near-copies of real SesameMulti photos
    z = np.load(META_DIR / "features.npz", allow_pickle=True)
    multi = pd.read_csv(SPLIT_DIR / "splits_multi.csv", keep_default_na=False)
    real = multi[~multi.is_synthetic.astype(str).str.lower().isin(["true", "1"])]
    pos = {u: i for i, u in enumerate(z["uids"])}
    ridx = [pos[u] for u in real.uid]
    R = z["feats"][ridx].astype(np.float32)
    cr = best_cos(feats, R)
    cand["max_cos_real"] = cr.max(1).round(4)
    cand["nearest_real_uid"] = real.uid.to_numpy()[cr.argmax(1)]
    near = cand[cand.max_cos_real >= NEAR_COPY]
    for _, r in near.iterrows():
        log(f"  near-copy of a real photo, dropped: {r['name']} ~ {r.nearest_real_uid} ({r.max_cos_real:.3f})")

    # near-identical synthetic tiles (same class): keep the first by name
    cs = best_cos(feats, feats)
    np.fill_diagonal(cs, -1.0)
    drop, dups = set(near.index), []
    for i in range(len(cand)):
        if i in drop:
            continue
        for j in np.where(cs[i] >= NEAR_COPY)[0]:
            if j > i and j not in drop and cand["cls"][i] == cand["cls"][j]:
                drop.add(int(j))
                dups.append(dict(kept=cand["name"][i], dropped=cand["name"][j], cos=round(float(cs[i, j]), 4)))
    for d in dups:
        log(f"  near-identical tiles: kept {d['kept']}, dropped {d['dropped']} ({d['cos']:.3f})")
    pool = cand.drop(index=sorted(drop))

    rng = np.random.default_rng(SEED)
    chosen = []
    for cls in SYN_CLASSES:
        g = pool[pool.cls == cls].sort_values("name")
        if len(g) < targets[cls]:
            if not args.allow_fewer:
                raise SystemExit(f"{cls}: only {len(g)} usable tiles, {targets[cls]} needed - "
                                 f"make {targets[cls] - len(g)} more, then run split and finalize again.")
            log(f"  {cls}: only {len(g)} usable tiles (target {targets[cls]}) - all of them are used (--allow-fewer)")
            chosen.append(g)
            continue
        chosen.append(g.iloc[np.sort(rng.choice(len(g), targets[cls], replace=False))])
    sel = pd.concat(chosen)

    rows = []
    for cls in SYN_CLASSES:
        s = sel[sel.cls == cls]
        batch = balanced_batches(s.groupby("raw_n").size().to_dict(), rng)
        out_dir = SYN2_ROOT / "trial_a_224" / f"{cls}{SYN_SUFFIX}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("syn2_*.jpg"):
            try:
                old.unlink()
            except OSError:
                log(f"  could not remove the old file {old.name} - delete it by hand")
        for i, r in s.iterrows():
            img = Image.open(GP_ROOT / r.tile_file).convert("RGB").resize((224, 224), Image.BICUBIC)
            path = out_dir / f"{r['name']}.jpg"
            img.save(path, quality=95)
            rows.append(dict(uid=make_uid(cls, path.name, synthetic=True), cls=cls, file=str(path.relative_to(GP_ROOT)),
                             tile_file=r.tile_file, raw_file=raw_file.get((cls, r.raw_n), ""), raw_n=r.raw_n,
                             pos=r.pos, tile_size=r.tile_size, group_id=f"SYN2-{CLASS_CODE[cls]}-{r.raw_n:04d}",
                             batch=batch[r.raw_n], max_cos_real=r.max_cos_real, nearest_real_uid=r.nearest_real_uid,
                             feature_row=int(i)))
    out = pd.DataFrame(rows)
    out.drop(columns="feature_row").to_csv(FINAL_TABLE, index=False, encoding="utf-8")
    cand.drop(columns=["md5"]).to_csv(META_DIR / "synthetic_v2_nearcopy.csv", index=False, encoding="utf-8")
    pd.DataFrame(dups, columns=["kept", "dropped", "cos"]).to_csv(META_DIR / "synthetic_v2_duplicates.csv", index=False)
    np.savez_compressed(META_DIR / "synthetic_v2_features.npz", uids=out.uid.to_numpy(),
                        feats=feats[out.feature_row.to_numpy()].astype(np.float16))
    log("\nFinal synthetic set:")
    log(out.groupby(["cls", "batch"]).size().unstack(fill_value=0).to_string())
    for cls in SYN_CLASSES:
        s = out[out.cls == cls]
        log(f"  {cls}: {len(s)} tiles from {s.raw_n.nunique()} Gemini images, highest similarity to a real photo "
            f"{s.max_cos_real.max():.3f} (near-copy threshold {NEAR_COPY})")
    log(f"Wrote {FINAL_TABLE}. Next:  python make_syn2_split.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["split", "finalize"])
    ap.add_argument("--redo", action="store_true", help="split: process every raw image again (same numbers)")
    ap.add_argument("--always-cover-logo", action="store_true", help="split: cover the corner on every image")
    ap.add_argument("--phyllody", type=int, default=75, help="finalize: number of Phyllody tiles to keep")
    ap.add_argument("--powdery", type=int, default=111, help="finalize: number of Powdery Mildew tiles to keep")
    ap.add_argument("--allow-fewer", action="store_true",
                    help="finalize: if a class has fewer usable tiles than asked for, use all of them instead of stopping")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    META_DIR.mkdir(parents=True, exist_ok=True)
    log = Tee(META_DIR / "process_synthetic_log.txt")
    log(f"=== process_synthetic.py {args.step}  {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    if args.step == "split":
        cmd_split(args, log)
    else:
        cmd_finalize(args, log)


if __name__ == "__main__":
    main()
