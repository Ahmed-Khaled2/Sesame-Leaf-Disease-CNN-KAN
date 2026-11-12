# Paper figures

Run these from the repository root. The figures are saved in `paper_figures/out/`.

| Script | Figure | Input |
|---|---|---|
| `figs_umap.py` | Fig. 2 (UMAP, field vs. lab vs. composited photos) and Fig. 8 (UMAP, real vs. synthetic) | `outputs/metadata/features.npz` (made by `build_metadata.py`) and `outputs/metadata/synthetic_v2_features.npz` (made by `process_synthetic.py finalize`); both are not included |
| `fig_mlp_head.py` | Fig. 3 (the MLP head) | none |
| `figs_results.py` | Fig. 5 (confusion matrices) and Fig. 6 (Grad-CAM) | `outputs/runs/*/confusion.csv` and `outputs/figures/` (Fig. 6 needs `gradcam_compare.png` from `gradcam.py`, not included) |
| `../kan_curves.py --paper-fig7` | Fig. 7 (learned functions of the last KAN layer, KAN and spline-only KAN) | the saved weights of two seed-0 runs (not included); the plotted data is in `outputs/figures/fig7_kan_curves_paper.npz` |

`figs_results.py` also draws an older layout of Fig. 7; the paper uses the one from `kan_curves.py --paper-fig7`.
Fig. 1 (overview of the study) and Fig. 4 (the KAN head, adapted from Liu et al.) are drawings, not made with code.
