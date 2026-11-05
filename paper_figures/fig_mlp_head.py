"""Fig. 3 of the paper: the MLP classifier head (2048 -> 128 ReLU -> 4). Fewer nodes are drawn than the real layer sizes.
Run from the repository root:  python paper_figures/fig_mlp_head.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "savefig.dpi": 500})
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "fig3_mlp_head.png")

fig = plt.figure(figsize=(3.3, 1.95))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 3.7); ax.set_ylim(-0.5, 1.68); ax.set_aspect("equal"); ax.axis("off")
X = [0.45, 1.85, 3.25]
inp = [1.38, 1.08, 0.42, 0.12]            # vertical dots in the gap = 2,048 inputs
hid = [1.38, 1.08, 0.42, 0.12]            # vertical dots in the gap = 128 hidden units
out = [1.29, 0.93, 0.57, 0.21]            # 4 classes
R = 0.12
edge = dict(color="#a0a0a0", lw=0.45, zorder=1)
for yi in inp:
    for yh in hid:
        ax.plot([X[0] + R, X[1] - R], [yi, yh], **edge)
for yh in hid:
    for yo in out:
        ax.plot([X[1] + R, X[2] - R], [yh, yo], **edge)


def node(x, y, fc="white"):
    ax.add_patch(plt.Circle((x, y), R, fc=fc, ec="black", lw=0.8, zorder=3))


for y in inp:
    node(X[0], y, "#eef3fb")
for y in hid:
    node(X[1], y)
    t = np.linspace(-1, 1, 30)                                   # ReLU glyph inside each hidden node
    ax.plot(X[1] + 0.07 * t, y - 0.04 + 0.08 * np.maximum(t, 0), color="black", lw=0.8, zorder=4)
for y, lab in zip(out, ["BB", "HL", "PI", "PM"]):
    node(X[2], y, "#fbefe9")
    ax.text(X[2], y, lab, ha="center", va="center", fontsize=5.2, zorder=4)
for x in (X[0], X[1]):
    for yy in (0.88, 0.75, 0.62):
        ax.plot(x, yy, "o", color="black", ms=1.3, zorder=3)
ax.text((X[0] + X[1]) / 2, 1.53, r"$W_1$", ha="center", va="center", fontsize=7.5)
ax.text((X[1] + X[2]) / 2, 1.53, r"$W_2$", ha="center", va="center", fontsize=7.5)
for x, a, b in ((X[0], "Input", "2,048 features"), (X[1], "Hidden layer", "128 units, ReLU"), (X[2], "Output", "4 classes")):
    ax.text(x, -0.15, a, ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(x, -0.35, b, ha="center", va="center", fontsize=6.4)
fig.savefig(OUT)
print("saved", OUT)
