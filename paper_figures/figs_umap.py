"""Fig. 2 and Fig. 8 of the paper: UMAP plots of ImageNet ResNet50 features of the final dataset.

Fig. 2 (Section III-A): field photos vs. the original MSLD laboratory photos vs. the same photos after background
compositing, one panel per class (Bacterial Blight, Healthy Leaf).
Fig. 8 (Section IV-G): real vs. Gemini-generated Phyllody and Powdery Mildew images, one joint UMAP.

Features: features.npz written by build_metadata.py (ResNet50, ImageNet weights, 224x224 input; view 0 = original
orientation). The same file holds the original laboratory photos (lab_files / lab_feats).
features.npz is not included in the repository (it is computed from the images); run build_metadata.py first.
Run from the repository root:
    python paper_figures/figs_umap.py
Needs: numpy, pandas, matplotlib, umap-learn (pip install umap-learn).
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "savefig.dpi": 500})
ap = argparse.ArgumentParser()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repository root
ap.add_argument("--meta", default=os.path.join(ROOT, "outputs", "metadata"))
ap.add_argument("--out", default=os.path.join(ROOT, "paper_figures", "out"))
ap.add_argument("--json", default=os.path.join(ROOT, "paper_figures", "out", "umap_distances.json"))
args = ap.parse_args()
META = args.meta.rstrip("/\\") + os.sep
FIG = args.out
os.makedirs(FIG, exist_ok=True)
SEED = 42

z = np.load(META + "features.npz", allow_pickle=True)
feat = dict(zip(z["uids"], z["feats"][:, 0, :].astype(np.float32)))          # view 0 = original orientation
lab_feat = dict(zip([str(p) for p in z["lab_files"]], z["lab_feats"][:, 0, :].astype(np.float32)))
m = pd.read_csv(META + "metadata.csv")
m = m[m.in_multi & ~m.excluded.fillna(False).astype(bool)]
print("rows used:", len(m), m.groupby(["source", "cls"]).size().to_dict())


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nn_dist(a, b):
    """cosine distance from each row of a to its nearest row of b"""
    return 1 - (l2(a) @ l2(b).T).max(axis=1)


def embed(x):
    return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(x)


def style(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("#666666")
    ax.set_xlabel("UMAP 1", fontsize=6.5, labelpad=2); ax.set_ylabel("UMAP 2", fontsize=6.5, labelpad=2)


# ------------------------------------------------------------------ Fig. A: background adaptation
C_FIELD, C_LAB, C_COMP = "#4d7f2f", "#d9822b", "#2e6db4"
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
check = {}
for ax, (cls, name, lab) in zip(axes, (("Bacteria Blight Infected", "Bacterial Blight", "(a)"),
                                     ("Healthy Leaf", "Healthy Leaf", "(b)"))):
    field = m[(m.cls == cls) & m.source.isin(["ERFD", "RSFD"]) & ~m.is_synthetic]
    comp = m[(m.cls == cls) & (m.source == "MSLD")]
    F_ = np.stack([feat[u] for u in field.uid])
    Cm = np.stack([feat[u] for u in comp.uid])
    La = np.stack([lab_feat[str(p)] for p in comp.lab_file])                     # the same photos before compositing
    e = embed(np.vstack([F_, La, Cm]))
    nf, nl = len(F_), len(La)
    ax.scatter(*e[:nf].T, s=3.2, c=C_FIELD, lw=0, alpha=0.75, label="Field photos (ERFD, RSFD)")
    ax.scatter(*e[nf:nf + nl].T, s=4, facecolors="none", edgecolors=C_LAB, lw=0.45, alpha=0.9, label="MSLD lab photos")
    ax.scatter(*e[nf + nl:].T, s=4.5, c=C_COMP, marker="x", lw=0.45, alpha=0.85, label="MSLD after compositing")
    style(ax)
    ax.set_title(f"{name}", fontsize=7, fontweight="bold", pad=3)
    ax.text(0.5, -0.17, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8, fontfamily="Liberation Serif")
    check[name] = dict(n_field=nf, n_lab=nl, lab_to_field=float(np.median(nn_dist(La, F_))),
                       comp_to_field=float(np.median(nn_dist(Cm, F_))),
                       centroid_lab=1 - (l2(La.mean(0, keepdims=True)) @ l2(F_.mean(0, keepdims=True)).T).item(),
                       centroid_comp=1 - (l2(Cm.mean(0, keepdims=True)) @ l2(F_.mean(0, keepdims=True)).T).item())
handles = [Line2D([], [], marker="o", ls="", color=C_FIELD, markersize=3.2, label="Field photos (ERFD, RSFD)"),
           Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=C_LAB, markersize=3.4,
                  markeredgewidth=0.7, label="MSLD lab photos"),
           Line2D([], [], marker="x", ls="", color=C_COMP, markersize=3.6, markeredgewidth=0.8,
                  label="MSLD after compositing")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.7, frameon=False, handletextpad=0.2, columnspacing=0.9)
fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.26, wspace=0.16)
fig.savefig(f"{FIG}/fig2_umap_background.png")
plt.close(fig)
print("background check:", check)
dist = {"bb_lab": check["Bacterial Blight"]["lab_to_field"], "bb_comp": check["Bacterial Blight"]["comp_to_field"],
        "hl_lab": check["Healthy Leaf"]["lab_to_field"], "hl_comp": check["Healthy Leaf"]["comp_to_field"]}

# ------------------------------------------------------------------ Fig. B: real vs synthetic
C_PI, C_PM = "#2e6db4", "#8e44ad"
groups = [("Phyllody Infected", False, "Phyllody, real"), ("Phyllody Infected", True, "Phyllody, synthetic"),
          ("Powdery Mildew", False, "Powdery Mildew, real"), ("Powdery Mildew", True, "Powdery Mildew, synthetic")]
X, idx = [], []
for cls, syn, name in groups:
    g = m[(m.cls == cls) & (m.is_synthetic == syn)]
    X.append(np.stack([feat[u] for u in g.uid])); idx.append(len(g))
e = embed(np.vstack(X))
fig, ax = plt.subplots(figsize=(3.5, 2.0))
start = 0
for (cls, syn, name), n in zip(groups, idx):
    col = C_PI if cls.startswith("Phyllody") else C_PM
    pts = e[start:start + n]
    if syn:
        ax.scatter(*pts.T, s=6, facecolors="none", edgecolors=col, lw=0.55, alpha=0.95, label=f"{name} ({n})")
    else:
        ax.scatter(*pts.T, s=5, c=col, lw=0, alpha=0.8, label=f"{name} ({n})")
    start += n
style(ax)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=5.8, frameon=False, handletextpad=0.2,
           markerscale=1.6, columnspacing=1.2)
fig.subplots_adjust(left=0.05, right=0.985, top=0.98, bottom=0.27)
fig.savefig(f"{FIG}/fig8_umap_synthetic.png")
plt.close(fig)
# share of synthetic images whose nearest neighbour (same class, real + synthetic pool) is another synthetic image
for cls in ("Phyllody Infected", "Powdery Mildew"):
    r = np.stack([feat[u] for u in m[(m.cls == cls) & ~m.is_synthetic].uid])
    s = np.stack([feat[u] for u in m[(m.cls == cls) & m.is_synthetic].uid])
    sim_ss = l2(s) @ l2(s).T; np.fill_diagonal(sim_ss, -1)
    nn_syn = sim_ss.max(1) > (l2(s) @ l2(r).T).max(1)
    dist[f"{cls[:2].lower()}_syn_nn_is_syn_pct"] = float(nn_syn.mean() * 100)
    print(cls, "synthetic whose NN is synthetic:", round(nn_syn.mean() * 100, 1), "% | median NN dist syn->real",
          round(float(np.median(nn_dist(s, r))), 3), "real->real",
          round(float(np.median(1 - np.sort(l2(r) @ l2(r).T, axis=1)[:, -2])), 3))
json.dump(dist, open(args.json, "w"), indent=1)
print("saved", FIG, args.json, dist)
