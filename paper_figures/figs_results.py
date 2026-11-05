"""Fig. 5, 6 and 7 of the paper, made from the files in outputs/.

  Fig. 5  confusion matrices on SesameMulti (KAN and MLP, test photos of all 3 seeds x 5 folds pooled)
          -> from outputs/runs/<run>/confusion.csv (included in the repository)
  Fig. 6  Grad-CAM, one test photo per class, KAN and MLP
          -> a crop of outputs/figures/gradcam_compare.png made by gradcam.py. That picture shows dataset photos,
             some of which may not be shared, so it is NOT included; run gradcam.py first (needs the images and the
             seed-0 weights, which train.py saves with --save-weights best).
  Fig. 7  learned functions of the last KAN layer, KAN and spline-only KAN
          -> crops of outputs/figures/kan_curves_*_last_layer.png made by kan_curves.py (included)

The crops use fixed pixel positions of the pictures written by gradcam.py and kan_curves.py.
Run from the repository root:   python paper_figures/figs_results.py
Needs: numpy, pandas, matplotlib, seaborn, pillow.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "savefig.dpi": 500})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "paper_figures", "out")
os.makedirs(FIG, exist_ok=True)
CLS = ["Bacteria Blight Infected", "Healthy Leaf", "Phyllody Infected", "Powdery Mildew"]
SHORT = ["BB", "HL", "PI", "PM"]


def fig5_confusion():
    runs = {"CNN-KAN": "P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s{}", "CNN-MLP": "P3_multi_mlp_h128_d0.2_ln_l4_r50_s{}"}
    cms = {}
    for name, pat in runs.items():
        cm = np.zeros((4, 4), int)
        for s in range(3):
            c = pd.read_csv(os.path.join(SRC, "runs", pat.format(s), "confusion.csv"))
            for _, r in c[c.split == "test"].iterrows():
                cm[CLS.index(r["true"]), CLS.index(r["pred"])] += r["count"]
        assert cm.sum() == 3 * 1606, cm.sum()
        cms[name] = cm
        print(name, "pooled test accuracy", round(np.trace(cm) / cm.sum() * 100, 2))
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95))
    for ax, (name, cmap), lab in zip(axes, (("CNN-KAN", "Purples"), ("CNN-MLP", "Oranges")), ("(a)", "(b)")):
        pctm = cms[name] / cms[name].sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{v:.1f}" if v >= 0.05 else "0" for v in row] for row in pctm])
        sns.heatmap(pctm, ax=ax, cmap=cmap, vmin=0, vmax=100, annot=annot, fmt="", cbar=False, square=True,
                    annot_kws={"fontsize": 6.3}, linewidths=0, xticklabels=SHORT, yticklabels=SHORT)
        ax.set_title(f"SesameMulti (4-class, unfrozen layer4)\n{name} Confusion Matrix", fontsize=5.3,
                     fontweight="bold", pad=4)
        ax.set_xlabel("Predicted", fontsize=6.5, labelpad=2)
        ax.set_ylabel("True", fontsize=6.5, labelpad=2)
        ax.tick_params(axis="both", labelsize=6.3, length=2, pad=1.5)
        ax.tick_params(axis="y", rotation=0)
        ax.text(0.5, -0.29, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8, fontfamily="Liberation Serif")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.85, bottom=0.25, wspace=0.42)
    fig.savefig(os.path.join(FIG, "fig5_confusion.png"))
    plt.close(fig)


def fig6_gradcam():
    png = os.path.join(SRC, "figures", "gradcam_compare.png")
    if not os.path.exists(png):
        print("Fig. 6 skipped: outputs/figures/gradcam_compare.png not found (run gradcam.py first)")
        return
    g = Image.open(png).convert("RGB")
    gx = [(20, 233), (256, 469), (491, 704), (727, 940)]
    gy = [(38, 250), (287, 499), (536, 748), (785, 998), (1035, 1247), (1284, 1496), (1533, 1745)]
    groups = [("BB", "ERFD"), ("BB", "MSLD"), ("BB", "RSFD"), ("HL", "ERFD"), ("HL", "MSLD"), ("PI", "ERFD"), ("PM", "RSFD")]
    gp = pd.read_csv(os.path.join(SRC, "figures", "gradcam_photos.csv"))
    code = dict(zip(CLS, SHORT))
    gp["c"] = gp.cls.map(code)
    gp["p"] = gp.predicted.map(code)
    grun = ["P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0", "P3_multi_mlp_h128_d0.2_ln_l4_r50_s0"]
    ref = {}
    for _, r in gp[gp.run == grun[0]].iterrows():
        ref.setdefault((r.c, r.source), r.uid)
    show = [("BB", "ERFD", "Bacterial Blight"), ("HL", "MSLD", "Healthy Leaf"), ("PI", "ERFD", "Phyllody"),
            ("PM", "RSFD", "Powdery Mildew")]
    row_labels = ["Test photo", "CNN-KAN", "CNN-MLP"]
    fig, axes = plt.subplots(3, 4, figsize=(3.5, 2.95))
    for j, (cl, src, longname) in enumerate(show):
        y0, y1 = gy[groups.index((cl, src))]
        for i in range(3):
            x0, x1 = gx[i]
            ax = axes[i, j]
            ax.imshow(g.crop((x0, y0, x1 + 1, y1 + 1)))
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if i == 0:
                ax.set_title(f"{longname}\n({src})", fontsize=6.2, pad=2.5)
            else:
                r = gp[(gp.run == grun[i - 1]) & (gp.uid == ref[(cl, src)])].iloc[0]
                ok = r.p == r.c
                ax.set_xlabel(f"{r.p} {r.prob:.2f}" + ("" if ok else " (wrong)"), fontsize=5.9, labelpad=1.2,
                              color="#2f5e0d" if ok else "#a32d2d")
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=6.4, fontweight="bold", labelpad=3)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.915, bottom=0.04, wspace=0.05, hspace=0.2)
    fig.savefig(os.path.join(FIG, "fig6_gradcam.png"))
    plt.close(fig)


def fig7_kan_curves():
    cols_x = [(63, 333), (396, 666), (729, 999), (1062, 1333)]
    rows_y = [(152, 326), (407, 581), (662, 836), (917, 1091)]

    def crop_panel(path, r, c=0):
        im = Image.open(path).convert("RGB")
        x0, x1 = cols_x[c]
        y0, y1 = rows_y[r]
        return im.crop((x0 - 48, y0 - 44, x1 + 6, y1 + 26))

    std = os.path.join(SRC, "figures", "kan_curves_P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0_last_layer.png")
    spo = os.path.join(SRC, "figures", "kan_curves_P4_multi_kan_h128_g5_o5_nobase_d0.2_ln_l4_r50_s0_last_layer.png")
    fig, axes = plt.subplots(4, 2, figsize=(3.5, 4.75))
    for c, (path, lab) in enumerate([(std, "KAN (base + spline)"), (spo, "Spline-only KAN")]):
        for r in range(4):
            axes[r, c].imshow(crop_panel(path, r, 0))
            axes[r, c].axis("off")
        axes[0, c].set_title(lab, fontsize=7.2, fontweight="bold", pad=3)
    handles = [Line2D([], [], color="#2C2C2A", lw=1.4, label="total edge function"),
               Line2D([], [], color="#888780", lw=1.0, ls="--", label="base part w·SiLU(x)"),
               Line2D([], [], color="#7F77DD", lw=1.2, label="spline part"),
               Patch(color="#D3D1C7", label="real inputs (test photos)"),
               Line2D([], [], color="#888780", lw=0.6, ls=":", label="end of spline support")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.8, frameon=False, handlelength=2.0, columnspacing=1.0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.075, wspace=0.03, hspace=0.04)
    fig.savefig(os.path.join(FIG, "fig7_kan_curves.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig5_confusion()
    fig6_gradcam()
    fig7_kan_curves()
    print("saved to", FIG)
