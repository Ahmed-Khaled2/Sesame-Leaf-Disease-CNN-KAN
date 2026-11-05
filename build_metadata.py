"""
build_metadata.py  -  STEP 1 of the journal re-run

Builds ONE table of every image (SesameMulti real + synthetic, and SesameBase)
with: class, source dataset, synthetic flag, Gemini parent image, photo time,
file hash, nearest look-alike, and a GROUP id.

A group = images that must always stay on the same side of a split.
Two real images are linked into the same group when:
  rule "copy"  : they are near-copies (any orientation: flips / 90-degree turns)
                 -> deep-feature similarity >= --copy-cos,
                    OR pHash distance <= --copy-phash together with similarity >= --copy-phash-cos
  rule "burst" : same source + same class + taken close together
                 (<= --burst-seconds apart, or camera numbers <= --burst-seq apart)
                 AND deep-feature similarity >= --burst-cos (--burst-cos-lab for MSLD lab photos)
Synthetic tiles cut from the same Gemini image share one group (train-only anyway).

Mendeley (MSLD) images were pasted onto random field backgrounds, so two photos of
the same leaf can look different only because of the background. For MSLD-vs-MSLD
pairs the check therefore uses the ORIGINAL white-background lab photos
(Combining Datasets/MSLD300_300). All other pairs use the training images.

Outputs (outputs/metadata/):
  metadata.csv              the image table (used by make_splits.py)
  metadata.xlsx             images, pairs, groups, cos_bins, summary, settings
  features.npz              saved features, so re-running with new thresholds is fast
  review/*.png              picture sheets to check the groups by eye
  build_metadata_log.txt

Run:                 python build_metadata.py
New thresholds:      python build_metadata.py --copy-cos 0.96 --burst-cos 0.88
(the second run re-uses features.npz, so it only takes seconds)
"""
import argparse
import datetime as dt
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import (GP_ROOT, MULTI_ROOT, BASE_ROOT, MSLD_LAB_ROOT, META_DIR, EXCLUDE_FILE, CLASSES, CLASS_CODE, SYN_SUFFIX,
                    list_images, strip_trial_prefix, make_uid, parse_source, parse_time,
                    parse_camera_seq, gemini_parent, file_md5, environment_info, Tee, set_all_seeds)


# ==========================================================================
# 1. Collect every image
# ==========================================================================
def collect_images(log):
    rows = {}

    def add(cls, folder, fname, synthetic, dataset):
        image_id = strip_trial_prefix(fname)
        uid = make_uid(cls, image_id, synthetic)
        path = Path(folder) / fname
        if uid not in rows:
            source = parse_source(image_id, synthetic)
            t = parse_time(image_id, source)
            rows[uid] = dict(
                uid=uid, cls=cls, source=source, is_synthetic=synthetic, image_id=image_id,
                file=str(path.relative_to(GP_ROOT)), in_multi=False, in_base=False,
                photo_time=t, camera_seq=parse_camera_seq(image_id, source),
                gemini_parent=gemini_parent(image_id) if synthetic else "",
            )
        rows[uid]["in_" + dataset] = True
        if dataset == "base":
            rows[uid]["file_base"] = str(path.relative_to(GP_ROOT))

    for cls in CLASSES:
        for f in list_images(MULTI_ROOT / cls):
            add(cls, MULTI_ROOT / cls, f, False, "multi")
        for f in list_images(MULTI_ROOT / (cls + SYN_SUFFIX)):
            add(cls, MULTI_ROOT / (cls + SYN_SUFFIX), f, True, "multi")
        for f in list_images(BASE_ROOT / cls):
            add(cls, BASE_ROOT / cls, f, False, "base")

    df = pd.DataFrame(list(rows.values()))

    # original lab photo for every MSLD composite ("IMG2025..._bg1.jpg" -> "Trial_A_IMG2025....jpg")
    lab_index = {}
    for cls in CLASSES:
        for f in list_images(MSLD_LAB_ROOT / cls):
            lab_index[(cls, Path(strip_trial_prefix(f)).stem)] = str((MSLD_LAB_ROOT / cls / f).relative_to(GP_ROOT))
    df["lab_file"] = [lab_index.get((c, i.split("_bg")[0]), "") if s == "MSLD" else ""
                      for c, i, s in zip(df.cls, df.image_id, df.source)]
    n_msld = int((df.source == "MSLD").sum())
    n_lab = int((df.lab_file != "").sum())
    log(f"  MSLD composites with their original lab photo found: {n_lab}/{n_msld}")
    if n_lab < n_msld:
        log("  WARNING: some MSLD lab photos are missing - those pairs are compared on the composites")

    if "file_base" not in df:
        df["file_base"] = ""
    df["file_base"] = df["file_base"].fillna("")
    df = df.sort_values(["is_synthetic", "cls", "source", "image_id"]).reset_index(drop=True)

    log(f"Images found: {len(df)}")
    log(f"  SesameMulti real : {int((df.in_multi & ~df.is_synthetic).sum())}")
    log(f"  Synthetic        : {int(df.is_synthetic.sum())}")
    log(f"  SesameBase       : {int(df.in_base.sum())}")
    unknown = df[df.source == "UNKNOWN"]
    if len(unknown):
        log(f"  WARNING: {len(unknown)} images with unknown source, e.g. {unknown.uid.head(5).tolist()}")
    return df


# ==========================================================================
# 2. Hashes and image features
# ==========================================================================
def load_rgb(path, size=224):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if (w, h) != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img, (w, h)


def phash_bits(img_rgb):
    """64-bit perceptual hash for each of the 8 orientations. Returns (8,) uint64."""
    from scipy.fft import dctn
    a = np.asarray(img_rgb.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    out = []
    for k in range(4):
        for flip in (False, True):
            b = np.rot90(a, k)
            if flip:
                b = np.fliplr(b)
            d = dctn(b, norm="ortho")[:8, :8]
            med = np.median(d.flatten()[1:])
            bits = (d > med).flatten()
            out.append(np.packbits(bits).view(">u8")[0])
    return np.array(out, dtype=np.uint64)


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint8)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1).astype(np.uint8)


def extract_features(files, args, log, what="images"):
    import torch
    from torchvision import models, transforms

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             ("cpu" if args.device == "auto" else args.device)
    log(f"Feature extractor: ResNet50 (ImageNet weights) on {device}")
    net = models.resnet50(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    n = len(files)
    feats = np.zeros((n, 8, 2048), dtype=np.float32)
    hashes = np.zeros((n, 8), dtype=np.uint64)
    md5s, sizes = [], []
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        views = []
        for k in range(4):
            r = torch.rot90(x, k, dims=(2, 3))
            views += [r, torch.flip(r, dims=(3,))]
        with torch.no_grad():
            f = net(torch.cat(views)).float()
        f = torch.nn.functional.normalize(f, dim=1).cpu().numpy()
        f = f.reshape(8, len(batch_imgs), 2048).transpose(1, 0, 2)
        feats[batch_idx] = f
        batch_imgs.clear()
        batch_idx.clear()

    for i, rel in enumerate(files):
        path = GP_ROOT / rel
        img, wh = load_rgb(path)
        sizes.append(f"{wh[0]}x{wh[1]}")
        md5s.append(file_md5(path))
        hashes[i] = phash_bits(img)
        batch_imgs.append(norm(img))
        batch_idx.append(i)
        if len(batch_imgs) == args.batch:
            flush()
        if (i + 1) % 200 == 0 or i + 1 == n:
            log(f"  features ({what}): {i + 1}/{n}")
    flush()
    return feats, hashes, md5s, sizes


# ==========================================================================
# 3. Similarity between every pair
# ==========================================================================
def similarity_matrices(feats, hashes, lab_idx=None, lab_feats=None, lab_hashes=None):
    """cos[i,j]  = best cosine between image i and any orientation of image j
       ham[i,j]  = smallest pHash distance over orientations"""
    f0 = feats[:, 0, :]
    cos = np.full((len(f0), len(f0)), -1.0, dtype=np.float32)
    ham = np.full((len(f0), len(f0)), 64, dtype=np.uint8)
    for t in range(8):
        cos = np.maximum(cos, f0 @ feats[:, t, :].T)
        ham = np.minimum(ham, popcount64(hashes[:, 0][:, None] ^ hashes[:, t][None, :]))
    cos = np.maximum(cos, cos.T)
    ham = np.minimum(ham, ham.T)
    if lab_idx is not None and len(lab_idx) > 1:
        # MSLD vs MSLD: compare the original lab photos instead of the composites
        lc, lh = similarity_matrices(lab_feats, lab_hashes)
        cos[np.ix_(lab_idx, lab_idx)] = lc
        ham[np.ix_(lab_idx, lab_idx)] = lh
    np.fill_diagonal(cos, -1.0)
    np.fill_diagonal(ham, 64)
    return cos, ham


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_pairs_and_groups(df, cos, ham, args, log):
    n = len(df)
    excluded = df["excluded"].to_numpy(dtype=bool) if "excluded" in df else np.zeros(n, dtype=bool)
    real = ~df.is_synthetic.values & ~excluded       # excluded images take no part in grouping
    cls = df.cls.to_numpy(dtype=object)
    src = df.source.to_numpy(dtype=object)
    times = df.photo_time.values
    lab_ok = (df.lab_file != "").to_numpy()
    seqs = df.camera_seq.values

    # --- candidate pairs: high similarity, small hash distance, or close in time
    cand = set()
    iu, ju = np.where(np.triu((cos >= args.pair_cos) | (ham <= args.pair_phash), k=1))
    cand.update(zip(iu.tolist(), ju.tolist()))

    time_pairs = {}
    by_key = defaultdict(list)
    for i in range(n):
        if real[i] and pd.notna(times[i]):
            by_key[(src[i], cls[i], "t")].append((pd.Timestamp(times[i]), i))
        if real[i] and pd.notna(seqs[i]):
            by_key[(src[i], cls[i], "s")].append((int(seqs[i]), i))
    for (s, c, kind), items in by_key.items():
        items.sort()
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                gap = items[b][0] - items[a][0]
                gap = gap.total_seconds() if kind == "t" else gap
                limit = args.burst_seconds if kind == "t" else args.burst_seq
                if gap > limit:
                    break
                i, j = sorted((items[a][1], items[b][1]))
                time_pairs[(i, j)] = (kind, gap)
                cand.add((i, j))

    # --- decide links
    uf = UnionFind(n)
    pair_rows = []
    for i, j in sorted(cand):
        both_real = real[i] and real[j]
        c, h = float(cos[i, j]), int(ham[i, j])
        kind, gap = time_pairs.get((i, j), (None, None))
        on_lab = bool(src[i] == "MSLD" and src[j] == "MSLD" and lab_ok[i] and lab_ok[j])
        rule = ""
        # pHash alone is not trusted (plain white lab backgrounds give low pHash distances
        # for different leaves), so a pHash match also needs a reasonable feature similarity.
        if both_real and (c >= args.copy_cos or (h <= args.copy_phash and c >= args.copy_phash_cos)):
            rule = "copy"
        elif both_real and kind is not None and c >= (args.burst_cos_lab if on_lab else args.burst_cos):
            rule = "burst"
        if rule:
            uf.union(i, j)
        pair_rows.append(dict(
            uid_a=df.uid[i], uid_b=df.uid[j], class_a=cls[i], class_b=cls[j],
            source_a=src[i], source_b=src[j], synthetic_a=bool(not real[i]), synthetic_b=bool(not real[j]),
            cos=round(c, 4), phash_dist=h,
            compared_on="lab photos" if on_lab else "training images",
            seconds_apart=gap if kind == "t" else None, camera_seq_gap=gap if kind == "s" else None,
            linked=bool(rule), rule=rule))
    pairs = pd.DataFrame(pair_rows)
    log(f"Candidate pairs checked: {len(pairs)} | linked: copy={int((pairs.rule == 'copy').sum()) if len(pairs) else 0}, "
        f"burst={int((pairs.rule == 'burst').sum()) if len(pairs) else 0}")

    # --- group ids: real groups named by size (G0001 = biggest), synthetic by Gemini parent
    roots = defaultdict(list)
    for i in range(n):
        if real[i]:
            roots[uf.find(i)].append(i)
    ordered = sorted(roots.values(), key=lambda m: (-len(m), df.uid[m[0]]))
    group_id = [""] * n
    for g, members in enumerate(ordered, start=1):
        for i in members:
            group_id[i] = f"G{g:04d}"
    for i in range(n):
        if excluded[i]:
            group_id[i] = f"EXCLUDED-{df.uid[i]}"
        elif not real[i]:
            group_id[i] = f"SYN-{CLASS_CODE[cls[i]]}-{df.gemini_parent[i]}"
    df["group_id"] = group_id
    df["group_size"] = df.groupby("group_id").uid.transform("count")

    rules_per_uid = defaultdict(set)
    if len(pairs):
        for r in pairs[pairs.linked].itertuples():
            rules_per_uid[r.uid_a].add(r.rule)
            rules_per_uid[r.uid_b].add(r.rule)
    df["linked_by"] = [",".join(sorted(rules_per_uid.get(u, []))) for u in df.uid]
    return df, pairs


def nearest_real_neighbour(df, cos, ham):
    real_idx = np.where(~df.is_synthetic.values)[0]
    sub = cos[:, real_idx]
    best = sub.argmax(1)
    nn = real_idx[best]
    df["nn_uid"] = df.uid.to_numpy(dtype=object)[nn]
    df["nn_class"] = df.cls.to_numpy(dtype=object)[nn]
    df["nn_cos"] = np.round(cos[np.arange(len(df)), nn], 4)
    df["nn_phash_dist"] = ham[np.arange(len(df)), nn]
    df["nn_same_class"] = df.nn_class == df.cls
    return df


# ==========================================================================
# 4. Summaries and review pictures
# ==========================================================================
def summaries(df, pairs, cos, args):
    real = df[~df.is_synthetic & ~df.excluded]
    out = {}
    out["counts_multi"] = pd.crosstab(real[real.in_multi].cls, real[real.in_multi].source, margins=True)
    out["counts_base"] = pd.crosstab(real[real.in_base].cls, real[real.in_base].source, margins=True)
    out["counts_synthetic"] = df[df.is_synthetic].groupby("cls").agg(
        images=("uid", "count"), gemini_images=("gemini_parent", "nunique"))

    g = real.groupby("group_id")
    sizes = g.uid.count()
    out["group_sizes"] = sizes.value_counts().sort_index().rename_axis("group_size").to_frame("n_groups")
    out["groups_by_class_source"] = real.assign(in_group=real.group_size > 1).groupby(["cls", "source"]).agg(
        images=("uid", "count"), images_in_multi_image_groups=("in_group", "sum"),
        groups=("group_id", "nunique"))

    # cosine histogram for real-real pairs, same class vs different class
    ridx = np.where(~df.is_synthetic.values)[0]
    sub = cos[np.ix_(ridx, ridx)]
    cl = df.cls.to_numpy(dtype=object)[ridx]
    so = df.source.to_numpy(dtype=object)[ridx]
    same = (cl[:, None] == cl[None, :])
    same_src = (so[:, None] == so[None, :])
    tri = np.triu(np.ones_like(same, dtype=bool), k=1)
    bins = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.01]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (sub >= lo) & (sub < hi) & tri
        rows.append(dict(cos_from=lo, cos_to=min(hi, 1.0),
                         same_class_same_source=int((m & same & same_src).sum()),
                         same_class_other_source=int((m & same & ~same_src).sum()),
                         different_class=int((m & ~same).sum())))
    out["cos_bins"] = pd.DataFrame(rows)

    cross = pairs[(pairs.linked) & (pairs.class_a != pairs.class_b)] if len(pairs) else pairs
    out["label_conflicts"] = cross.sort_values("cos", ascending=False)
    gc = real.groupby(["group_id", "cls"]).uid.count().reset_index(name="n")
    gc["share"] = gc.n / gc.cls.map(real.cls.value_counts())
    biggest_share = float(gc.share.max()) if len(gc) else 0.0
    syn = df[df.is_synthetic]
    out["checks"] = pd.DataFrame([
        dict(check="exact duplicate files (same md5)", value=int(df.duplicated("md5", keep=False).sum())),
        dict(check="linked pairs across DIFFERENT classes (label conflict?)", value=len(cross)),
        dict(check="groups containing more than one class", value=int((g.cls.nunique() > 1).sum())),
        dict(check="largest group size", value=int(sizes.max())),
        dict(check="largest share of one class inside one group (should stay small, e.g. < 3%)",
             value=f"{100 * biggest_share:.1f}%"),
        dict(check="images excluded by exclude.csv", value=int(df.excluded.sum()) if "excluded" in df else 0),
        dict(check="synthetic images whose nearest REAL image has cos >= copy threshold",
             value=int((syn.nn_cos >= args.copy_cos).sum())),
        dict(check="synthetic max cos to any real image", value=f"{float(syn.nn_cos.max()):.4f}" if len(syn) else ""),
        dict(check="images with unknown source", value=int((df.source == "UNKNOWN").sum())),
    ])
    out["checks"]["value"] = out["checks"]["value"].astype(str)

    return out


def review_sheets(df, pairs, cos, out_dir, args, log, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    uid2row = {u: i for i, u in enumerate(df.uid)}
    thumbs = {}

    def thumb(uid, lab=False):
        row = uid2row[uid]
        rel = df.lab_file[row] if (lab and df.lab_file[row]) else df.file[row]
        if rel not in thumbs:
            img = Image.open(GP_ROOT / rel).convert("RGB")
            img.thumbnail((160, 160))
            thumbs[rel] = img
        return thumbs[rel]

    def short(uid):
        s = uid.split("/", 1)[1]
        return (s[:22] + "..") if len(s) > 23 else s

    def pair_sheet(sel, title, fname):
        if len(sel) == 0:
            return
        sel = sel.head(30)
        rows = len(sel)
        fig, axes = plt.subplots(rows, 2, figsize=(7.5, 3.0 * rows))
        axes = np.atleast_2d(axes)
        for r, p in enumerate(sel.itertuples()):
            lab = getattr(p, "compared_on", "") == "lab photos"
            for c, (u, k, s) in enumerate([(p.uid_a, p.class_a, p.source_a), (p.uid_b, p.class_b, p.source_b)]):
                ax = axes[r, c]
                ax.imshow(thumb(u, lab))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{CLASS_CODE[k]} {s}{' (lab photo)' if lab else ''} {short(u)}", fontsize=7)
            extra = []
            if pd.notna(p.seconds_apart):
                extra.append(f"{p.seconds_apart:.0f}s apart")
            if pd.notna(p.camera_seq_gap):
                extra.append(f"cam# gap {int(p.camera_seq_gap)}")
            axes[r, 1].set_xlabel(f"cos={p.cos:.3f}  pHash={p.phash_dist}  {' '.join(extra)}  "
                                  f"-> {'LINKED (' + p.rule + ')' if p.linked else 'not linked'}", fontsize=7)
        H = 3.0 * rows
        fig.suptitle(title, fontsize=10, y=1 - 0.3 / H)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / H))
        fig.savefig(out_dir / fname, dpi=90)
        plt.close(fig)
        log(f"  review sheet: {fname}")

    rp = pairs[~pairs.synthetic_a & ~pairs.synthetic_b] if len(pairs) else pairs
    if len(rp):
        # pairs by similarity band, so the thresholds can be judged by eye
        bands = [(0.99, 1.01), (0.98, 0.99), (0.97, 0.98), (0.96, 0.97), (0.94, 0.96), (0.92, 0.94), (0.90, 0.92)]
        for lo, hi in bands:
            sel = rp[(rp.cos >= lo) & (rp.cos < hi)]
            sel = sel.sample(min(len(sel), 30), random_state=seed).sort_values("cos", ascending=False)
            pair_sheet(sel, f"Real pairs with cos {lo:.2f}-{min(hi, 1):.2f}  ({len(rp[(rp.cos >= lo) & (rp.cos < hi)])} pairs in total, random 30 shown)",
                       f"pairs_cos_{int(lo * 100)}_{int(min(hi, 1) * 100)}.png")
        pair_sheet(rp[rp.rule == "burst"].sample(min(30, int((rp.rule == "burst").sum())), random_state=seed),
                   "Linked by BURST rule (random 30)", "linked_burst_sample.png")
        pair_sheet(rp[(rp.seconds_apart.notna() | rp.camera_seq_gap.notna()) & ~rp.linked]
                   .sort_values("cos", ascending=False),
                   "Close in time but NOT linked (most similar first) - should any of these be linked?",
                   "time_close_not_linked.png")
        pair_sheet(rp[(~rp.linked)].sort_values("cos", ascending=False),
                   "Most similar pairs that are NOT linked - should any of these be linked?",
                   "most_similar_not_linked.png")
        pair_sheet(rp[rp.linked & (rp.class_a != rp.class_b)],
                   "Linked pairs with DIFFERENT labels - possible label errors", "cross_class_links.png")

    # largest groups
    real = df[~df.is_synthetic]
    big = real.groupby("group_id").uid.apply(list)
    big = big[big.apply(len) > 1].sort_values(key=lambda s: s.apply(len), ascending=False).head(15)
    if len(big):
        cols = 10
        fig, axes = plt.subplots(len(big), cols, figsize=(cols * 1.5, len(big) * 1.7))
        axes = np.atleast_2d(axes)
        for r, (gid, members) in enumerate(big.items()):
            for c in range(cols):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                if c < len(members):
                    ax.imshow(thumb(members[c]))
                    ax.set_title(short(members[c])[:14], fontsize=6)
                else:
                    ax.axis("off")
            axes[r, 0].set_ylabel(f"{gid}\n{len(members)} imgs", fontsize=7)
        H = len(big) * 1.7
        fig.suptitle("Largest groups (first 10 images of each)", fontsize=10, y=1 - 0.25 / H)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / H))
        fig.savefig(out_dir / "largest_groups.png", dpi=80)
        plt.close(fig)
        log("  review sheet: largest_groups.png")

    # synthetic -> nearest real image
    syn = df[df.is_synthetic].sort_values("nn_cos", ascending=False).head(20)
    if len(syn):
        sel = pd.DataFrame(dict(uid_a=syn.uid.to_numpy(dtype=object), uid_b=syn.nn_uid.to_numpy(dtype=object),
                                class_a=syn.cls.to_numpy(dtype=object), class_b=syn.nn_class.to_numpy(dtype=object), source_a="GEMINI",
                                source_b=df.source.to_numpy(dtype=object)[[uid2row[u] for u in syn.nn_uid]],
                                cos=syn.nn_cos.values, phash_dist=syn.nn_phash_dist.values,
                                seconds_apart=None, camera_seq_gap=None, linked=False, rule=""))
        pair_sheet(sel, "Synthetic images and their closest REAL image (a copy would mean leakage)",
                   "synthetic_vs_real.png")


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-cos", type=float, default=0.97, help="near-copy: feature similarity threshold")
    ap.add_argument("--copy-phash", type=int, default=6, help="near-copy: max pHash distance (of 64 bits)")
    ap.add_argument("--burst-seconds", type=float, default=60, help="burst: max seconds between photos")
    ap.add_argument("--burst-seq", type=int, default=3, help="burst: max camera-number gap (IMG_xxxx names)")
    ap.add_argument("--copy-phash-cos", type=float, default=0.95,
                    help="near-copy by pHash also needs at least this feature similarity")
    ap.add_argument("--burst-cos", type=float, default=0.90, help="burst: min feature similarity (field photos)")
    ap.add_argument("--burst-cos-lab", type=float, default=0.95,
                    help="burst: min similarity for MSLD lab photos (different leaves on white paper look alike)")
    ap.add_argument("--pair-cos", type=float, default=0.90, help="pairs listed in the Excel/review if cos >= this")
    ap.add_argument("--pair-phash", type=int, default=12, help="...or pHash distance <= this")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recompute", action="store_true", help="ignore saved features.npz")
    ap.add_argument("--no-review", action="store_true", help="skip the review pictures")
    args = ap.parse_args()

    set_all_seeds(0)
    META_DIR.mkdir(parents=True, exist_ok=True)
    log_path = META_DIR / "build_metadata_log.txt"
    log = Tee(log_path)
    log("=" * 70)
    log(f"build_metadata.py  started {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"GP folder: {GP_ROOT}")
    log("Settings: " + ", ".join(f"{k}={v}" for k, v in vars(args).items()))

    for p in (MULTI_ROOT, BASE_ROOT):
        if not p.is_dir():
            raise SystemExit(f"Folder not found: {p}")

    df = collect_images(log)
    df["excluded"] = False
    df["exclude_reason"] = ""
    if EXCLUDE_FILE.exists():
        ex = pd.read_csv(EXCLUDE_FILE, keep_default_na=False, comment="#")
        ex = ex[ex.uid.astype(str).str.strip() != ""]
        unknown = sorted(set(ex.uid) - set(df.uid))
        if unknown:
            log(f"  WARNING: exclude.csv lists uids that do not exist: {unknown}")
        reasons = dict(zip(ex.uid, ex.reason))
        df["excluded"] = df.uid.isin(reasons)
        df["exclude_reason"] = df.uid.map(reasons).fillna("")
        log(f"  excluded by exclude.csv: {int(df.excluded.sum())} images")

    feat_file = META_DIR / "features.npz"
    lab_idx = np.where(df.lab_file != "")[0]
    lab_files = df.lab_file.to_numpy(dtype=object)[lab_idx].tolist()
    z = np.load(feat_file, allow_pickle=True) if (feat_file.exists() and not args.recompute) else None
    if z is not None and list(z["uids"]) == df.uid.tolist() and list(z["lab_files"]) == lab_files:
        log("Re-using saved features.npz (use --recompute to rebuild)")
        feats, hashes, md5s, sizes = z["feats"], z["hashes"], list(z["md5s"]), list(z["sizes"])
        lab_feats, lab_hashes = z["lab_feats"], z["lab_hashes"]
    else:
        if z is not None:
            log("features.npz does not match the current images -> recomputing")
        feats, hashes, md5s, sizes = extract_features(df.file.tolist(), args, log, "training images")
        if len(lab_files):
            lab_feats, lab_hashes, _, _ = extract_features(lab_files, args, log, "MSLD lab photos")
        else:
            lab_feats, lab_hashes = np.zeros((0, 8, 2048), np.float32), np.zeros((0, 8), np.uint64)
    np.savez_compressed(feat_file, uids=np.array(df.uid.tolist()), feats=feats.astype(np.float16),
                        hashes=hashes, md5s=np.array(md5s), sizes=np.array(sizes),
                        lab_files=np.array(lab_files, dtype=object), lab_feats=lab_feats.astype(np.float16),
                        lab_hashes=lab_hashes)
    feats = feats.astype(np.float32)
    lab_feats = lab_feats.astype(np.float32)
    df["md5"] = md5s
    df["original_size"] = sizes
    dup_first = df.groupby("md5").uid.transform("first")
    df["exact_duplicate_of"] = np.where(dup_first != df.uid, dup_first, "")

    cos, ham = similarity_matrices(feats, hashes, lab_idx, lab_feats, lab_hashes)
    df, pairs = build_pairs_and_groups(df, cos, ham, args, log)
    df = nearest_real_neighbour(df, cos, ham)
    summ = summaries(df, pairs, cos, args)

    real = df[~df.is_synthetic & ~df.excluded]
    log(f"Real images: {len(real)} -> {real.group_id.nunique()} groups "
        f"({int((real.group_size > 1).sum())} images share a group with another image)")
    for _, r in summ["checks"].iterrows():
        log(f"  check | {r['check']}: {r['value']}")

    # ---------------- save
    df_out = df.copy()
    df_out["photo_time"] = df_out.photo_time.astype(str).replace({"NaT": "", "None": ""})
    df_out.to_csv(META_DIR / "metadata.csv", index=False, encoding="utf-8")

    groups = real.groupby("group_id").agg(
        size=("uid", "count"), n_classes=("cls", "nunique"),
        classes=("cls", lambda s: "; ".join(sorted(set(s)))),
        sources=("source", lambda s: "; ".join(sorted(set(s)))),
        linked_by=("linked_by", lambda s: "; ".join(sorted(set(",".join(s).split(",")) - {""}))),
        members=("uid", lambda s: " | ".join(s)[:30000])).reset_index().sort_values("size", ascending=False)

    settings = pd.DataFrame(
        [dict(setting=k, value=str(v)) for k, v in vars(args).items()] +
        [dict(setting="GP folder", value=str(GP_ROOT)),
         dict(setting="SesameMulti folder", value=str(MULTI_ROOT)),
         dict(setting="SesameBase folder", value=str(BASE_ROOT))] +
        [dict(setting="env." + k, value=str(v)) for k, v in environment_info().items()])

    with pd.ExcelWriter(META_DIR / "metadata.xlsx", engine="openpyxl") as xw:
        df_out.to_excel(xw, sheet_name="images", index=False)
        pairs.to_excel(xw, sheet_name="pairs", index=False)
        groups.to_excel(xw, sheet_name="groups", index=False)
        summ["cos_bins"].to_excel(xw, sheet_name="cos_bins", index=False)
        summ["checks"].to_excel(xw, sheet_name="checks", index=False)
        summ["label_conflicts"].to_excel(xw, sheet_name="label_conflicts", index=False)
        summ["counts_multi"].to_excel(xw, sheet_name="counts_multi")
        summ["counts_base"].to_excel(xw, sheet_name="counts_base")
        summ["counts_synthetic"].to_excel(xw, sheet_name="counts_synthetic")
        summ["group_sizes"].to_excel(xw, sheet_name="group_sizes")
        summ["groups_by_class_source"].to_excel(xw, sheet_name="groups_by_class_source")
        settings.to_excel(xw, sheet_name="settings", index=False)
    log(f"Saved: {META_DIR / 'metadata.csv'}")
    log(f"Saved: {META_DIR / 'metadata.xlsx'}")

    if not args.no_review:
        review_sheets(df, pairs, cos, META_DIR / "review", args, log)
    log(f"Done {dt.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
