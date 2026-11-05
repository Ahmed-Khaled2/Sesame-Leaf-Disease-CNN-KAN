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

Output (outputs/figures/):
    kan_curves_<run>_last_layer.png    per class, the 4 strongest connections into it
    kan_curves_<run>_first_layer.png   the 12 strongest connections of the first KAN layer
    kan_curves_edges.csv               every connection: effect, spline share (all runs)
    kan_curves_summary.csv             per run and layer: spline share, inputs inside the grid
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
    cache = train.ImageCache(test.file.tolist(), print)
    loader = DataLoader(train.EvalSet(cache, test.file.tolist(), test.label.tolist()), batch_size=64)
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


def run_one(run, fold, dev, edges_rows, summary_rows):
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
        Xl = X[li].to(dev)
        eff, mb, ms = edge_stats(layer, Xl)
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
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    edges, summary = [], []
    for run in args.runs:
        print(f"KAN curves for {run}")
        try:
            run_one(run, args.fold, dev, edges, summary)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
    if summary:
        pd.DataFrame(edges).to_csv(FIG_DIR / "kan_curves_edges.csv", index=False)
        pd.DataFrame(summary).to_csv(FIG_DIR / "kan_curves_summary.csv", index=False)
        print(pd.DataFrame(summary).to_string(index=False))
    else:
        print("Nothing to draw (no KAN run with saved weights found).")


if __name__ == "__main__":
    main()
