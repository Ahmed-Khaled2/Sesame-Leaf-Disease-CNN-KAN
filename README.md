# Sesame Leaf Disease Classification with a CNN-KAN Head

Code, data splits, results and synthetic images for the paper:

> A. Khaled et al., "Sesame Leaf Disease Detection Using a Hybrid CNN-KAN Architecture," manuscript under review, 2026.

The study compares a Kolmogorov-Arnold Network (KAN) classification head with MLP and linear heads on a pretrained
ResNet50 backbone, for four sesame leaf classes: Bacterial Blight, Healthy Leaf, Phyllody and Powdery Mildew.
All results come from held-out test folds of group-aware 5-fold cross-validation, repeated with 3 seeds.

## Main results

Test accuracy (%), mean ± standard deviation over 15 fold-runs (3 seeds x 5 folds), ResNet50 with its last block
(layer4) fine-tuned:

| Head | SesameBase (3 classes) | SesameMulti (4 classes) |
|---|---|---|
| KAN (128 hidden, grid 5, order 5) | 95.6 ± 1.9 | 92.3 ± 1.1 |
| MLP (128 hidden) | 96.3 ± 1.9 | 92.4 ± 1.2 |
| MLP (1,536 hidden, same size as the KAN head) | 95.4 ± 1.6 | 92.3 ± 1.1 |
| Linear | 95.8 ± 1.9 | 92.2 ± 1.3 |
| KAN without LayerNorm | 96.5 ± 1.8 | 92.7 ± 1.3 |
| Spline-only KAN | 96.2 ± 1.8 | 92.9 ± 1.9 |

With ResNet50, no difference between two heads is statistically significant (paired, corrected resampled t-test;
all p-values above 0.3). All tables, paired tests and equivalence tests are in
[`outputs/all_runs.xlsx`](outputs/all_runs.xlsx).

## What is in this repository

| Path | Contents |
|---|---|
| `common.py` | Paths, class names and helpers shared by all scripts |
| `build_metadata.py` | Step 1: one table of all images, with near-duplicate groups |
| `make_splits.py` | Step 2: the fixed group-aware 5-fold splits |
| `make_extra_splits.py` | Extra splits for the leakage check and the source check |
| `train.py` | Step 3: one run = one dataset, one head, one seed, all 5 folds |
| `run_queue.py` | Runs the lines of a phase file, split over several terminals |
| `phases/` | The exact runs of each phase (phase 3 is the main comparison of the paper) |
| `summarize.py` | Step 4: collects all runs into `outputs/all_runs.xlsx`, with the statistics |
| `kan_curves.py`, `gradcam.py`, `bench_speed.py` | Learned KAN functions, Grad-CAM maps, measured speed (they share `analysis_common.py`) |
| `exclude.csv` | Images left out of every experiment, with the reason |
| `outputs/` | Our results: image table, splits, all 143 runs and the summary workbook |
| `synthetic_images/` | The 186 Gemini-generated training images |
| `prompts/PROMPTS.md` | The prompts used to generate them |
| `dataset_construction/` | Notebooks used to build the image folders from the source datasets |
| `paper_figures/` | Scripts for Figs. 2, 3, 5, 6, 7 and 8 of the paper |

## Data

The source photos are not included here. They come from these datasets:

| Dataset | Classes used | Where to get it |
|---|---|---|
| ERFD, part 1: Nibret et al. [1] | Bacterial Blight, Healthy Leaf, Phyllody | Not public. It was obtained from the original author and cannot be shared. |
| ERFD, part 2: Hailu et al. [2] | Bacterial Blight, Healthy Leaf, Phyllody | With the paper: https://doi.org/10.1038/s41598-025-08076-1 |
| MSLD: Rahman and Hena [3] | Bacterial Blight, Healthy Leaf | https://doi.org/10.17632/c64jt5gkzm.1 |
| RSFD: three Roboflow datasets [4]-[6] | Bacterial Blight, Powdery Mildew | https://universe.roboflow.com/new-workspace-mwunv/sesame, https://universe.roboflow.com/test-mrio3/sesame-bgkdo, https://universe.roboflow.com/essa-ahmed-3dkth/sesame-diseases |
| Synthetic (this repository) | Phyllody, Powdery Mildew (training only) | [`synthetic_images/`](synthetic_images) |

Dataset versions used in the paper (number of images):

| Version | Bacterial Blight | Healthy Leaf | Phyllody | Powdery Mildew | Total |
|---|---|---|---|---|---|
| SesameBase (ERFD) | 233 | 216 | 106 | - | 555 |
| SesameMulti | 827 | 515 | 106 | 158 | 1,606 |
| SesameMulti+Syn | 827 | 515 | 181 | 269 | 1,792 |

### Folder layout

The scripts read the images from this layout (the notebooks in `dataset_construction/` build it):

```text
<data root>/
  Sesame Datasets/
    Combining Datasets/
      SesameMulti Dataset + Synthetic Data/
        Bacteria Blight Infected/              Trial_A_<file name>
        Healthy Leaf/
        Phyllody Infected/
        Powdery Mildew/
        Phyllody Infected - Synthetic/         copy of synthetic_images/trial_a_224/...
        Powdery Mildew - Synthetic/
      MSLD300_300/<class>/                     original MSLD lab photos (only used to find near-duplicates)
    Sesame_Preprocessing_Study/
      Trial_A (Resize)/CombinedPaper_Clean/<class>/                                   SesameBase
      Trial_B (Resize + Contrast Stretching)/CombinedPaper_Clean/<class>/             phase 1 only
      Trial_C (Resize + Contrast Stretching + CLAHE)/CombinedPaper_Clean/<class>/
      Trial_D (Resize + Contrast Stretching + CLAHE + Median Filter)/CombinedPaper_Clean/<class>/
```

By default the data root is the folder that contains this repository. To use another folder, set the environment
variable `SESAME_GP_ROOT`. The `file` column of `outputs/splits/splits_*.csv` lists every image path, and
`outputs/metadata/metadata.csv` has the MD5 hash of every image, so a rebuilt folder can be checked against ours.

## Setup

```bash
python -m venv .venv
# activate it, then install PyTorch for your system (see https://pytorch.org), for example:
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

We used Python 3.11.9, PyTorch 2.9.0 with CUDA 13.0, torchvision 0.24.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.3 and
scikit-learn 1.7.2, on Windows with one NVIDIA GeForce RTX 5060 GPU. The KAN layers come from
[efficient-kan](https://github.com/Blealtan/efficient-kan).

## How to reproduce the results

1. **Image table and near-duplicate groups:** `python build_metadata.py`
   writes `outputs/metadata/` (our `metadata.csv` is already included).
2. **Splits:** `python make_splits.py` and `python make_extra_splits.py`.
   The splits we used are already in `outputs/splits/`, and both scripts refuse to overwrite them unless you pass
   `--force`. The split seed is 42.
3. **Training:** each line of a phase file is one `train.py` run. Start one worker per terminal:

   ```bash
   python run_queue.py phases/phase3.txt --worker 0 --workers 3
   python run_queue.py phases/phase3.txt --worker 1 --workers 3
   python run_queue.py phases/phase3.txt --worker 2 --workers 3
   ```

   A single run by hand: `python train.py --data multi --head kan --seed 0`.
   The training seeds are 0, 1 and 2. Phase 3 (54 runs) took about 10 hours with 3 workers on one RTX 5060.

   | Phase file | Runs |
   |---|---|
   | `phase0.txt` | One fold, to check the setup and the time per fold |
   | `phase1.txt` | Preprocessing trials A to D and backbones, SesameBase, frozen backbone (chosen on validation data) |
   | `phase2.txt` | Head settings on SesameMulti, 8 KAN and 8 MLP settings (chosen on validation data) |
   | `phase3.txt` | The main comparison: all heads, 3 seeds, SesameBase, SesameMulti and SesameMulti+Syn |
   | `phase4.txt` | Extra checks: EfficientNet-B0, source check, less training data, random splits, figures and speed |

   `outputs/runs/` already holds our 143 runs, and finished runs are skipped. To train again, move
   `outputs/runs/` somewhere else first (keep `outputs/splits/`).
4. **Summary and statistics:** `python summarize.py` writes `outputs/all_runs.xlsx`.
5. **Analyses:** `python kan_curves.py` and `python gradcam.py` need the saved weights of the seed-0 runs, which are
   not included (about 2 GB); `phase3.txt` and `phase4.txt` save them with `--save-weights best`.
   `python bench_speed.py` measures the speed with ImageNet weights, so it needs no trained weights.
6. **Paper figures:** `python paper_figures/fig_mlp_head.py`, `python paper_figures/figs_results.py` and
   `python paper_figures/figs_umap.py` (this one needs `outputs/metadata/features.npz` from step 1).
   The figures are saved in `paper_figures/out/`.

## Result files

- **`outputs/all_runs.xlsx`** has one sheet per table: `runs`, `folds`, `phase1_choice`, `phase2_choice`,
  `by_config` (the tables of the paper), `comparisons` (paired head-vs-head tests: mean difference, 95% confidence
  interval, the corrected resampled t-test of Nadeau and Bengio, and the TOST equivalence test),
  `other_comparisons` (with vs. without synthetic images, frozen vs. fine-tuned), `kan_check`, `kan_knockout`,
  `speed` and `kan_curves`.
- **`outputs/runs/<run name>/`** has the files of one run: `config.json` (all settings, library versions, and
  fingerprints of the split file and the code), `folds.csv` (scores per fold), `epochs.csv` (every epoch),
  `per_class.csv`, `confusion.csv`, `summary.csv`, `kan_check.csv` and `kan_knockout.csv` (KAN runs), and
  `run_log.xlsx` (all tables in one file, including the prediction for every test photo and the class balancing of
  every epoch).
- **Run names** read like this: `P3_multi_kan_h128_g5_o5_d0.2_ln_l4_r50_s0` = phase 3, SesameMulti, KAN head with 128
  hidden units, grid 5, order 5, dropout 0.2, LayerNorm, layer4 fine-tuned, ResNet50, seed 0. Other parts:
  `base-A` = SesameBase with preprocessing trial A, `nobase` = spline-only KAN, `none` = no LayerNorm, `frozen` = frozen backbone, `effb0` = EfficientNet-B0,
  `rand` = photo-level random split, `f25`/`f50` = 25% or 50% of the training photos, `bbhl_cross` = the source check.
- **`outputs/metadata/metadata.csv`** has one row per image: class, source, synthetic flag, the Gemini image a tile
  was cut from, MD5 hash, near-duplicate group (`group_id`) and nearest look-alike.
- **`outputs/splits/`**: `splits_multi.csv` and `splits_base.csv` are the main splits; `*_random.csv` are for the
  leakage check and `splits_bbhl_*.csv` for the source check. The columns `fold0` to `fold4` give the role of each
  image (train, val or test) in each fold.

The result files are the ones written by the runs. The only change: in the split tables, the `\` in the image paths
was replaced by `/`, so the code also runs on Linux and macOS. The split fingerprints (computed from the `uid` and
fold columns) did not change. The logs show the local folder of the computer that ran them.

## Synthetic images

[`synthetic_images/`](synthetic_images) holds 75 Phyllody and 111 Powdery Mildew images, generated on 11 April 2026
in the Gemini app with Nano Banana 2 (Gemini 3.1 Flash Image). They were used in training folds only, never for
validation or testing. `original/` has the tiles as kept after cleaning; `trial_a_224/` has the same tiles resized to
224 x 224, exactly as used for training. The images were checked visually by the first author, but not by a plant
pathologist. The prompts are in [`prompts/PROMPTS.md`](prompts/PROMPTS.md).

## Dataset construction notebooks

These notebooks built the image folders. They still have the Windows paths of the computer they ran on, so change the
paths before running them.

| Notebook | Step |
|---|---|
| `Change data to class.ipynb` | Turns the Roboflow object-detection datasets into class folders: each image gets the class with the largest total box area. Images without boxes were checked by hand and relabeled. |
| `Cropping hands out.ipynb` | Finds hands with YOLO11 (COCO person class) [7] and crops them out of the photos |
| `Compare duplicates.ipynb` | Finds duplicates within and between folders with a perceptual hash |
| `Check for augmentation.ipynb` | Finds flipped or augmented copies with ResNet50 feature similarity |
| `Full dataset image preprocessing.ipynb` | Preprocessing trials A to D (resize to 224 x 224, contrast stretching, CLAHE, median filter) |
| `Smart Image Selection.ipynb` | Diversity-aware selection: ResNet50 features, UMAP and K-Means, then even sampling from each cluster |
| `Adding background to MSLD.ipynb` | Puts the MSLD lab leaves on field backgrounds with RMBG-1.4 [8], with UMAP, colour and FID checks |

## Citation

If you use this code or data, please cite the paper:

```bibtex
@unpublished{khaled2026sesame,
  title  = {Sesame Leaf Disease Detection Using a Hybrid {CNN-KAN} Architecture},
  author = {Khaled, Ahmed and others},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

- Code: [MIT](LICENSE).
- Synthetic images, prompts, splits, image table and results: [CC BY 4.0](DATA_LICENSE.md).
- The source datasets keep their own licenses and are not included.

## Acknowledgment

Thanks to Eyerusalem Alebachew Nibret for providing the ERFD dataset.

## References

1. E. A. Nibret, A. M. Mequanenit, A. M. Ayalew, K. Kusrini, and R. Martínez-Béjar, "Sesame plant disease
   classification using deep convolution neural networks," *Appl. Sci.*, vol. 15, no. 4, Art. no. 2124, 2025.
2. A. A. Hailu, B. C. Kassa, E. A. Desta, F. B. Adugna, and A. O. Salau, "Ensemble-based sesame disease detection and
   classification using deep convolutional neural networks (CNN)," *Sci. Rep.*, vol. 15, Art. no. 28757, 2025.
3. S. A. Rahman and M. H. Hena, "Applying convolutional neural networks for early detection of diseases in sesame
   leaf," Mendeley Data, V1, 2025.
4. "Sesame Dataset," Roboflow Universe, workspace new-workspace-mwunv, 2022.
5. "Sesame Dataset," Roboflow Universe, workspace test-mrio3, 2022.
6. E. Ahmed, "Sesame Diseases Dataset," Roboflow Universe, 2025.
7. G. Jocher and J. Qiu, "Ultralytics YOLO11," 2024. https://github.com/ultralytics/ultralytics
8. BRIA AI, "RMBG-1.4: Background removal model," Hugging Face, 2024.
