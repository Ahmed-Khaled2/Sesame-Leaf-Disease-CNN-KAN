# Dataset construction notebooks

These notebooks built the image folders used by the training code, from the source datasets listed in the main
README. They still have the Windows paths of the computer they ran on, so change the paths before running them.

| Notebook | Step |
|---|---|
| `Change data to class.ipynb` | Turns the Roboflow object-detection datasets into class folders: each image gets the class with the largest total box area. Images without boxes were checked by hand and relabeled. |
| `Cropping hands out.ipynb` | Finds hands with YOLO11 (COCO person class) and crops them out of the photos |
| `Compare duplicates.ipynb` | Finds duplicates within and between folders with a perceptual hash |
| `Check for augmentation.ipynb` | Finds flipped or augmented copies with ResNet50 feature similarity |
| `Full dataset image preprocessing.ipynb` | Preprocessing trials A to D (resize to 224 x 224, contrast stretching, CLAHE, median filter) |
| `Smart Image Selection.ipynb` | Diversity-aware selection: ResNet50 features, UMAP and K-Means, then even sampling from each cluster |
| `Adding background to MSLD.ipynb` | Puts the MSLD lab leaves on field backgrounds with RMBG-1.4, with UMAP, colour and FID checks |

The near-duplicate groups and the splits used in the paper were made later by `build_metadata.py` and
`make_splits.py` in the repository root.
