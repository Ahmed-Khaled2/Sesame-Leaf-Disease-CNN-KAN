# Synthetic images

75 Phyllody and 111 Powdery Mildew images, generated on 11 April 2026 in the Gemini app (Google) with
Nano Banana 2 (Gemini 3.1 Flash Image). The prompts are in [`../prompts/PROMPTS.md`](../prompts/PROMPTS.md).

| Folder | Contents |
|---|---|
| `original/<class>/` | The tiles as kept after cleaning, at full size |
| `trial_a_224/<class> - Synthetic/` | The same tiles resized to 224 x 224 (preprocessing trial A), exactly as used for training |

- Most Gemini outputs were 2 x 2 grids that were cut into four tiles. The 75 Phyllody tiles come from 18 generated
  images and the 111 Powdery Mildew tiles from 62. The column `gemini_parent` of
  `outputs/metadata/metadata.csv` gives the source image of every tile, and tiles of the same source image share
  one near-duplicate group.
- They were used in training folds only (dataset SesameMulti+Syn), never for validation or testing.
- They were checked visually by the first author, but not by a plant pathologist, so they may contain biological
  errors.
- To train with them, copy the two folders in `trial_a_224/` into
  `Sesame Datasets/Combining Datasets/SesameMulti Dataset + Synthetic Data/` (see the main README).

License: CC BY 4.0 (see [`../DATA_LICENSE.md`](../DATA_LICENSE.md)).
