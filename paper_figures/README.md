# Paper figures

Run these from the repository root. The figures are saved in `paper_figures/out/`.

| Script | Figure | Input |
|---|---|---|
| `figs_umap.py` | Fig. 2 (UMAP, field vs. lab vs. composited photos) and Fig. 8 (UMAP, real vs. synthetic) | `outputs/metadata/features.npz`, made by `build_metadata.py` (not included) |
| `fig_mlp_head.py` | Fig. 3 (the MLP head) | none |
| `figs_results.py` | Fig. 5 (confusion matrices), Fig. 6 (Grad-CAM) and Fig. 7 (learned KAN functions) | `outputs/runs/*/confusion.csv` and `outputs/figures/` (Fig. 6 needs `gradcam_compare.png` from `gradcam.py`, not included) |

Fig. 1 (the pipeline) and Fig. 4 (the KAN head, adapted from Liu et al.) are drawings, not made with code.
