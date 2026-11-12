# Synthetic images

75 Phyllody and 111 Powdery Mildew images, generated on 9 November 2026 in the Gemini app (Google), with the model
setting "Auto (Gemini 3 models)", from text prompts alone: no photo was uploaded. The prompts are in
[`../prompts/PROMPTS.md`](../prompts/PROMPTS.md).

| Folder | Contents |
|---|---|
| `original/<class>/` | The tiles at full size, as kept after the visual check |
| `trial_a_224/<class> - Synthetic/` | The same tiles resized to 224 x 224 (preprocessing trial A), exactly as used for training |

- Every Gemini output was a 2 x 2 grid that was cut into four tiles (`process_synthetic.py`). The 75 Phyllody tiles
  come from 28 Gemini images and the 111 Powdery Mildew tiles from 39. A tile name gives its Gemini image and its
  position: `syn2_PI_0004_tr` = Phyllody image 4, top right. `outputs/metadata/synthetic_v2.csv` lists every tile with
  its Gemini image, and [`../prompts/image_prompts.csv`](../prompts/image_prompts.csv) gives the prompt of every
  Gemini image.
- No tile is a near-copy of a real photo: the highest cosine similarity of ResNet50 features to any real photo is
  0.92 for Phyllody and 0.95 for Powdery Mildew (the near-copy threshold is 0.97).
- They are used in training folds only (dataset `multi_syn2`), never for validation or testing. The Gemini images are
  split into 5 batches, and fold k trains on the 4 batches other than batch k (`make_syn2_split.py`), so the
  synthetic images stay below half of each class's training images in every fold.
- They were checked visually, but not by a plant pathologist, so they may contain biological errors.
- To train with them, copy the two folders in `trial_a_224/` into `Sesame Datasets/Synthetic Data v2/trial_a_224/`
  (see the main README).

License: CC BY 4.0 (see [`../DATA_LICENSE.md`](../DATA_LICENSE.md)).
