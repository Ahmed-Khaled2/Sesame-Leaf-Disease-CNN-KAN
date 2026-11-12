"""
kan_curves.py - what the KAN layers learned, drawn over the inputs they really receive.

In efficient-kan every connection is   phi(x) = w_base * SiLU(x) + spline(x).
For each KAN run with saved weights (fold 0 by default) this script:
  * captures the real inputs of every KAN layer on that fold's TEST photos,
  * measures each connection's effect (mean |phi(x)| over those inputs) and its spline share,
  * draws the strongest connections: total curve (black), base part w*SiLU (gray dashed),
    spline part (purple), with a histogram of the real inputs behind them (light gray).
    Dotted vertical lines = where the spline grid ends (outside it the spline part is zero).

Default runs: standard KAN and spline-only KAN, SesameMulti and SesameBase, seed 0.
    python kan_curves.py
    python kan_curves.py --runs P4_multi_kan_h128_g5_o5_nobase_d0.2_ln_l4_r50_s0 --fold 2
Per-class spline share for every fold of the seed-0 KAN runs (no figures, nothing else overwritten):
    python kan_curves.py --class-shares-only --folds 0,1,2,3,4 --runs P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0 P3_base-A_kan_h128_g5_o5_d0.2_ln_l4_r50_s0

Fig. 7 of the paper, drawn at its printed size (full page width; KAN on top, spline-only KAN below, the strongest
connection into each class of the last KAN layer, SesameMulti seed 0, fold 0, test photos):
    python kan_curves.py --paper-fig7
    -> outputs/figures/fig7_kan_curves_paper.png and fig7_kan_curves_paper.npz (the plotted data); nothing else is written

Output (outputs/figures/):
    kan_curves_<run>_last_layer.png    per class, the 4 strongest connections into it
    kan_curves_<run>_first_layer.png   the 12 strongest connections of the first KAN layer
    kan_curves_edges.csv               every connection: effect, spline share (all runs)
    kan_curves_summary.csv             per run and layer: spline share, inputs inside the grid
    kan_curves_class_shares.csv        per run, fold and class: share of what the class output receives
                                       that comes from the splines (last KAN layer, test photos)
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
from analysis_common import FIG_DIR, run_args, load_model, short, device
from torch.utils.data import DataLoader

DEFAULT_RUNS = ["P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0",
                "P4_multi_kan_h128_g5_o5_nobase_d0.2_ln_l4_r50_s0",
                "P3_base-A_kan_h128_g5_o5_d0.2_ln_l4_r50_s0",
                "P4_base-A_kan_h128_g5_o5_nobase_d0.2_ln_l4_r50_s0"]
PURPLE, GRAY, BLACK, HIST = "#7F77DD", "#888780", "#2C2C2A", "#D3D1C7"


@torch.no_grad()
def capture_inputs(model, sp, fold, dev):
    """Inputs of every KAN layer on the test photos of this fold."""
    test = sp[sp[f"fold{fold}"] == "test"]
    files = [f.replace("\\", "/") for f in test.file]      # split files may hold Windows paths
    cache = train.ImageCache(files, print)
    loader = DataLoader(train.EvalSet(cache, files, test.label.tolist()), batch_size=64)
    store = {i: [] for i in range(len(model.kan_layers))}
    hooks = [layer.register_forward_pre_hook(lambda m, inp, i=i: store[i].append(inp[0].detach().float()))
             for i, layer in enumerate(model.kan_layers)]
    for x, _ in loader:
        model(x.to(dev))
    for h in hooks:
        h.remove()
    return {i: torch.cat(v) for i, v in store.items()}


@torch.no_grad()
def edge_stats(layer, X, chunk=8):
    """effect[j, i] = mean |phi_ji(x_i)|, plus mean |base| and mean |spline| per connection."""
    B = layer.b_splines(X)                                   # (N, in, coeff)
    S = layer.scaled_spline_weight                           # (out, in, coeff)
    act = layer.base_activation(X)                           # (N, in)
    eff, mb, ms = [], [], []
    for j0 in range(0, layer.out_features, chunk):
        s = torch.einsum("nic,jic->nji", B, S[j0:j0 + chunk])
        b = act[:, None, :] * layer.base_weight[j0:j0 + chunk][None]
        eff.append((s + b).abs().mean(0))
        mb.append(b.abs().mean(0))
        ms.append(s.abs().mean(0))
    return torch.cat(eff).cpu().numpy(), torch.cat(mb).cpu().numpy(), torch.cat(ms).cpu().numpy()


@torch.no_grad()
def curve(layer, i, j, xs):
    pts = torch.zeros(len(xs), layer.in_features, device=layer.base_weight.device)
    pts[:, i] = torch.as_tensor(xs, dtype=pts.dtype, device=pts.device)
    spline = (layer.b_splines(pts)[:, i, :] @ layer.scaled_spline_weight[j, i, :]).cpu().numpy()
    base = (layer.base_activation(pts[:, i]) * layer.base_weight[j, i]).cpu().numpy()
    return base, spline


def panel(ax, layer, X, i, j, title, share):
    xi = X[:, i].cpu().numpy()
    g0, g1 = float(layer.grid[i, 0]), float(layer.grid[i, -1])
    lo, hi = min(np.percentile(xi, 0.5), g0) - 0.3, max(np.percentile(xi, 99.5), g1) + 0.3
    xs = np.linspace(lo, hi, 300)
    base, spline = curve(layer, i, j, xs)
    ax2 = ax.twinx()
    ax2.hist(xi, bins=40, range=(lo, hi), color=HIST, alpha=0.8)
    ax2.set_yticks([])
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(xs, base + spline, color=BLACK, lw=1.6, label="total")
    ax.plot(xs, base, color=GRAY, lw=1.1, ls="--", label="base (w x SiLU)")
    ax.plot(xs, spline, color=PURPLE, lw=1.3, label="spline")
    for g in (g0, g1):
        ax.axvline(g, color=GRAY, lw=0.6, ls=":")
    ax.set_title(f"{title}\nspline share {share:.0%}", fontsize=8)
    ax.tick_params(labelsize=7)


def draw_layer(layer, X, eff, mb, ms, pairs, titles, ncols, path, suptitle):
    n = len(pairs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.3), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (j, i), t in zip(axes.ravel(), pairs, titles):
        share = ms[j, i] / max(ms[j, i] + mb[j, i], 1e-12)
        panel(ax, layer, X, i, j, t, share)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


PAPER_FIG7_RUNS = [("P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0", "KAN (base + spline)"),
                   ("P4_multi_kan_h128_g5_o5_nobase_d0.2_ln_l4_r50_s0", "Spline-only KAN")]
CLASS_TITLE = {"Bacteria Blight Infected": "Bacterial Blight", "Healthy Leaf": "Healthy Leaf",
               "Phyllody Infected": "Phyllody", "Powdery Mildew": "Powdery Mildew"}


def paper_fig7(dev, fold=0):
    """Fig. 7 at its printed size: 2 rows (KAN, spline-only KAN) x 4 columns (classes), 7.1 in wide, 7 pt text."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = []
    for run, label in PAPER_FIG7_RUNS:
        a, cfg, run_dir = run_args(run)
        sp, classes, *_ = train.load_split(a)
        model = load_model(a, len(classes), run_dir, fold, dev)
        if model is None:
            raise SystemExit(f"{run}: no saved weights for fold {fold}")
        X = capture_inputs(model, sp, fold, dev)
        last = len(model.kan_layers) - 1
        layer, Xl = model.kan_layers[last], X[last].to(dev)
        eff, mb, ms = edge_stats(layer, Xl)
        row = []
        for j, c in enumerate(classes):                  # the strongest connection into each class
            i = int(np.argmax(eff[j]))
            xi = Xl[:, i].cpu().numpy()
            g0, g1 = float(layer.grid[i, 0]), float(layer.grid[i, -1])
            lo, hi = min(np.percentile(xi, 0.5), g0) - 0.3, max(np.percentile(xi, 99.5), g1) + 0.3
            xs = np.linspace(lo, hi, 400)
            base, spline = curve(layer, i, j, xs)
            row.append(dict(cls=c, hidden=i, xi=xi, xs=xs, base=base, spline=spline, g0=g0, g1=g1,
                            share=float(ms[j, i] / max(ms[j, i] + mb[j, i], 1e-12))))
            print(f"  {label}: hidden {i} -> {short(c)}, spline share {row[-1]['share']:.1%}")
        rows.append((label, row))

    rc = {"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 7, "axes.linewidth": 0.6,
          "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "mathtext.default": "regular"}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.45))
        for r, (label, row) in enumerate(rows):
            for c, pnl in enumerate(row):
                ax = axes[r, c]
                ax2 = ax.twinx()
                ax2.hist(pnl["xi"], bins=40, range=(pnl["xs"][0], pnl["xs"][-1]), color=HIST, alpha=0.9)
                ax2.set_yticks([])
                for side in ("top", "right"):
                    ax2.spines[side].set_visible(False)
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
                if r == 0:                               # spline-only KAN: the total is the spline itself
                    ax.plot(pnl["xs"], pnl["base"] + pnl["spline"], color=BLACK, lw=1.3)
                    ax.plot(pnl["xs"], pnl["base"], color=GRAY, lw=0.9, ls="--")
                ax.plot(pnl["xs"], pnl["spline"], color=PURPLE, lw=1.1)
                for g in (pnl["g0"], pnl["g1"]):
                    ax.axvline(g, color=GRAY, lw=0.5, ls=":")
                ax.tick_params(labelsize=6, pad=1.5)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.set_title(f"hidden unit {pnl['hidden']}, spline share {pnl['share']:.0%}", fontsize=6.3, pad=2.5,
                             color="#444441")
                if r == 0:
                    ax.annotate(CLASS_TITLE.get(pnl["cls"], pnl["cls"]), xy=(0.5, 1.21), xycoords="axes fraction",
                                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
                if r == 1:
                    ax.set_xlabel("input value", fontsize=6.3, labelpad=1.5)
            axes[r, 0].set_ylabel(label, fontsize=7.2, fontweight="bold", labelpad=4)
        handles = [Line2D([], [], color=BLACK, lw=1.3, label="total function"),
                   Line2D([], [], color=GRAY, lw=0.9, ls="--", label=r"base part $w_b$·SiLU(x)"),
                   Line2D([], [], color=PURPLE, lw=1.1, label="spline part"),
                   Patch(color=HIST, label="inputs on the test photos"),
                   Line2D([], [], color=GRAY, lw=0.5, ls=":", label="end of spline support")]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5, frameon=False, handlelength=2.2,
                   columnspacing=1.6)
        fig.subplots_adjust(left=0.075, right=0.99, top=0.885, bottom=0.17, wspace=0.28, hspace=0.58)
        out = FIG_DIR / "fig7_kan_curves_paper.png"
        fig.savefig(out, dpi=500)
        plt.close(fig)
    np.savez(FIG_DIR / "fig7_kan_curves_paper.npz", **{f"{r}_{k}_{key}": np.asarray(v)
             for r, (label, row) in enumerate(rows) for k, pnl in enumerate(row)
             for key, v in pnl.items() if key != "cls"})
    print(f"  saved {out}")


def run_one(run, fold, dev, edges_rows, summary_rows, class_rows, figures=True):
    a, cfg, run_dir = run_args(run)
    if a.head != "kan":
        print(f"  {run} is not a KAN run - skipped")
        return
    sp, classes, *_ = train.load_split(a)
    model = load_model(a, len(classes), run_dir, fold, dev)
    if model is None:
        print(f"  {run}: no saved weights for fold {fold} - skipped")
        return
    X = capture_inputs(model, sp, fold, dev)
    kind = "spline-only KAN" if a.kan_base == "off" else ("KAN, GP head (no LayerNorm)" if a.norm == "none" else "KAN")
    last = len(model.kan_layers) - 1
    for li, layer in enumerate(model.kan_layers):
        if not figures and li != last:
            continue                         # class shares need only the last layer
        Xl = X[li].to(dev)
        eff, mb, ms = edge_stats(layer, Xl)
        if li == last:                       # per class: sum over the connections into the class output
            for j, c in enumerate(classes):
                s, b = float(ms[j].sum()), float(mb[j].sum())
                class_rows.append(dict(run=run, fold=fold, cls=c, n_test_images=int(Xl.size(0)),
                                       sum_mean_abs_spline=s, sum_mean_abs_base=b,
                                       spline_share=s / max(s + b, 1e-12)))
            s, b = float(ms.sum()), float(mb.sum())
            class_rows.append(dict(run=run, fold=fold, cls="all classes", n_test_images=int(Xl.size(0)),
                                   sum_mean_abs_spline=s, sum_mean_abs_base=b, spline_share=s / max(s + b, 1e-12)))
        if not figures:
            continue
        inside = ((Xl > layer.grid[:, 0]) & (Xl < layer.grid[:, -1])).float().mean().item()
        summary_rows.append(dict(run=run, fold=fold, layer=li + 1, connections=eff.size,
                                 spline_share_of_output=float(ms.sum() / max(ms.sum() + mb.sum(), 1e-12)),
                                 median_spline_share_per_connection=float(np.median(ms / np.maximum(ms + mb, 1e-12))),
                                 inputs_inside_spline_grid=inside))
        jj, ii = np.unravel_index(np.argsort(-eff, axis=None), eff.shape)
        top = min(2000, eff.size)
        for j, i in zip(jj[:top], ii[:top]):
            edges_rows.append(dict(run=run, fold=fold, layer=li + 1, input=int(i), output=int(j),
                                   effect=float(eff[j, i]), mean_abs_base=float(mb[j, i]),
                                   mean_abs_spline=float(ms[j, i])))
        if li == last:                       # per class: the 4 strongest connections into it
            pairs, titles = [], []
            for j, c in enumerate(classes):
                order = np.argsort(-eff[j])[:4]
                pairs += [(j, int(i)) for i in order]
                titles += [f"hidden {int(i)} -> {short(c)}" for i in order]
            draw_layer(layer, Xl, eff, mb, ms, pairs, titles, 4, FIG_DIR / f"kan_curves_{run}_last_layer.png",
                       f"{kind}: last KAN layer, 4 strongest connections per class (fold {fold}, test photos)")
        if li == 0 and last > 0:             # first layer: 12 strongest connections
            pairs = [(int(j), int(i)) for j, i in zip(jj[:12], ii[:12])]
            titles = [f"feature {i} -> hidden {j}" for j, i in pairs]
            draw_layer(layer, Xl, eff, mb, ms, pairs, titles, 4, FIG_DIR / f"kan_curves_{run}_first_layer.png",
                       f"{kind}: first KAN layer, 12 strongest connections (fold {fold}, test photos)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", default="", help="several folds, e.g. 0,1,2,3,4 (overrides --fold)")
    ap.add_argument("--paper-fig7", action="store_true",
                    help="only Fig. 7 of the paper at its printed size (fold 0 of the two SesameMulti seed-0 KAN runs)")
    ap.add_argument("--class-shares-only", action="store_true",
                    help="only the per-class spline shares: no figures, edges or summary files are written")
    args = ap.parse_args()
    folds = [int(f) for f in args.folds.split(",") if f.strip()] or [args.fold]
    figures = not args.class_shares_only
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    if args.paper_fig7:
        paper_fig7(dev, args.fold)
        return
    edges, summary, class_rows = [], [], []
    for run in args.runs:
        for fold in folds:
            print(f"KAN curves for {run}, fold {fold}")
            try:
                run_one(run, fold, dev, edges, summary, class_rows, figures=figures)
            except FileNotFoundError as e:
                print(f"  skipped: {e}")
    if class_rows:
        out = FIG_DIR / "kan_curves_class_shares.csv"
        new = pd.DataFrame(class_rows)
        if out.exists():                     # keep rows of other runs/folds from earlier calls
            old = pd.read_csv(out)
            old = old[~old.set_index(["run", "fold"]).index.isin(new.set_index(["run", "fold"]).index)]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(["run", "fold", "cls"]).to_csv(out, index=False)
        piv = pd.DataFrame(class_rows).pivot_table(index=["run", "fold"], columns="cls", values="spline_share")
        print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    if figures and summary:
        pd.DataFrame(edges).to_csv(FIG_DIR / "kan_curves_edges.csv", index=False)
        pd.DataFrame(summary).to_csv(FIG_DIR / "kan_curves_summary.csv", index=False)
        print(pd.DataFrame(summary).to_string(index=False))
    elif not class_rows:
        print("Nothing to draw (no KAN run with saved weights found).")


if __name__ == "__main__":
    main()
