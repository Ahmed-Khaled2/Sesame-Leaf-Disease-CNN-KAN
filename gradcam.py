"""
gradcam.py - Grad-CAM heatmaps on TEST photos only. Each photo is explained by the model of the
fold where that photo was a test photo, so the model never saw it during training.

What it is for: checking WHERE the model looks (the leaf and its spots, or the background /
photo style). It is not a KAN-vs-MLP test - the heatmap is taken from the backbone, before the head.

Default runs (they have saved weights): seed-0 KAN and MLP on SesameMulti (phase 3) and the
spline-only KAN (phase 4).
    python gradcam.py
    python gradcam.py --runs P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0 --per-group 4

Output (outputs/figures/):
    gradcam_<run>.png     rows = class + source, each photo twice: plain and with the heatmap
    gradcam_compare.png   one photo per class + source, side by side for all runs
    gradcam_photos.csv    every photo: true class, prediction, probability, fold
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train
from analysis_common import (FIG_DIR, run_args, load_model, load_photo, pick_photos, short, device)

DEFAULT_RUNS = ["P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0",
                "P3_multi_mlp_h128_d0.2_ln_l4_r50_s0",
                "P4_multi_kan_h128_g5_o5_nobase_d0.2_ln_l4_r50_s0"]


def grad_cam(model, x):
    """Returns (cam 224x224 in [0,1], probabilities)."""
    store = {}

    def fwd_hook(m, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = model.cam_layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            x = x.unsqueeze(0).requires_grad_(True)
            out = model(x)
            prob = F.softmax(out, 1)[0].detach().cpu().numpy()
            model.zero_grad(set_to_none=True)
            out[0, int(prob.argmax())].backward()
    finally:
        h.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = (cam / cam.max()).cpu().numpy() if float(cam.max()) > 0 else cam.cpu().numpy()
    return cam, prob


def run_one(run_name, per_group, dev):
    a, cfg, run_dir = run_args(run_name)
    sp, classes, *_ = train.load_split(a)
    photos = pick_photos(sp, per_group)
    models, rows = {}, []
    for p in photos:
        k = p["fold"]
        if k not in models:
            models[k] = load_model(a, len(classes), run_dir, k, dev)
            if models[k] is None:
                print(f"  {run_name}: no saved weights for fold {k} - skipped")
        if models[k] is None:
            continue
        img, x = load_photo(p["file"])
        cam, prob = grad_cam(models[k], x.to(dev))
        pred = classes[int(prob.argmax())]
        rows.append(dict(run=run_name, **p, predicted=pred, prob=float(prob.max()),
                         correct=pred == p["cls"], img=img, cam=cam))
    return rows


def draw_grid(rows, title, path, per_group):
    groups = sorted({(r["cls"], r["source"]) for r in rows})
    fig, axes = plt.subplots(len(groups), 2 * per_group, figsize=(2 * per_group * 1.9, len(groups) * 2.05),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for gi, g in enumerate(groups):
        rs = [r for r in rows if (r["cls"], r["source"]) == g]
        for j, r in enumerate(rs[:per_group]):
            a0, a1 = axes[gi, 2 * j], axes[gi, 2 * j + 1]
            a0.imshow(r["img"])
            a1.imshow(r["img"])
            a1.imshow(r["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            a1.set_title(f"pred {short(r['predicted'])} {r['prob']:.2f}", fontsize=8,
                         color="#3B6D11" if r["correct"] else "#A32D2D")
            if j == 0:
                a0.set_title(f"{short(g[0])} | {g[1]}", fontsize=9, loc="left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.18 / len(groups)))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def draw_compare(all_rows, runs, path):
    by_run = {r: [x for x in all_rows if x["run"] == r] for r in runs}
    runs = [r for r in runs if by_run[r]]
    if len(runs) < 2:
        return
    groups = sorted({(x["cls"], x["source"]) for x in by_run[runs[0]]})
    first = {g: next(x for x in by_run[runs[0]] if (x["cls"], x["source"]) == g) for g in groups}
    fig, axes = plt.subplots(len(groups), 1 + len(runs), figsize=((1 + len(runs)) * 2.0, len(groups) * 2.1),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for gi, g in enumerate(groups):
        ref = first[g]
        axes[gi, 0].imshow(ref["img"])
        axes[gi, 0].set_title(f"{short(g[0])} | {g[1]}", fontsize=9, loc="left")
        for ri, run in enumerate(runs):
            hit = [x for x in by_run[run] if x["uid"] == ref["uid"]]
            if not hit:
                continue
            x = hit[0]
            ax = axes[gi, ri + 1]
            ax.imshow(x["img"])
            ax.imshow(x["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.set_title(f"{label(run)}: {short(x['predicted'])} {x['prob']:.2f}", fontsize=8,
                         color="#3B6D11" if x["correct"] else "#A32D2D")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def label(run):
    if "nobase" in run:
        return "spline-only KAN"
    if "_kan_" in run:
        return "KAN" if "_none_" not in run else "KAN (GP)"
    if "_mlp_" in run:
        return "MLP"
    return "linear" if "_linear_" in run else run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--per-group", type=int, default=3, help="photos per class + source")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    all_rows = []
    for run in args.runs:
        print(f"Grad-CAM for {run}")
        try:
            rows = run_one(run, args.per_group, dev)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        if not rows:
            continue
        all_rows += rows
        draw_grid(rows, f"Grad-CAM, test photos only - {label(run)} ({run})", FIG_DIR / f"gradcam_{run}.png",
                  args.per_group)
    if not all_rows:
        print("Nothing to draw (no saved weights found).")
        return
    draw_compare(all_rows, args.runs, FIG_DIR / "gradcam_compare.png")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("img", "cam")} for r in all_rows]).to_csv(
        FIG_DIR / "gradcam_photos.csv", index=False)
    acc = pd.DataFrame(all_rows).groupby("run").correct.mean()
    print("share of the shown photos predicted correctly:", acc.round(3).to_dict())


if __name__ == "__main__":
    main()
