"""
common.py - shared settings for the journal re-run.

Every script in this folder imports this file, so paths, class names and
helper functions are defined in ONE place only.

Folder layout this file expects (it finds everything relative to itself):

    Everything GP related/
        Journal Rerun/              <- this folder (scripts + outputs)
        Sesame Datasets/
            Combining Datasets/SesameMulti Dataset + Synthetic Data/
            Sesame_Preprocessing_Study/Trial_A (Resize)/CombinedPaper_Clean/
            ...
"""
import os
import re
import sys
import json
import random
import hashlib
import platform
import datetime as dt
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# "Everything GP related". Can be overridden for testing with an env variable.
GP_ROOT = Path(os.environ.get("SESAME_GP_ROOT", str(HERE.parent)))
DATASETS = GP_ROOT / "Sesame Datasets"

MULTI_ROOT = DATASETS / "Combining Datasets" / "SesameMulti Dataset + Synthetic Data"
# Original white-background Mendeley photos (before compositing). Used only to
# check whether two composited images show the same leaf.
MSLD_LAB_ROOT = DATASETS / "Combining Datasets" / "MSLD300_300"

PREPROC_ROOT = DATASETS / "Sesame_Preprocessing_Study"
PREPROC_TRIALS = {
    "A": PREPROC_ROOT / "Trial_A (Resize)" / "CombinedPaper_Clean",
    "B": PREPROC_ROOT / "Trial_B (Resize + Contrast Stretching)" / "CombinedPaper_Clean",
    "C": PREPROC_ROOT / "Trial_C (Resize + Contrast Stretching + CLAHE)" / "CombinedPaper_Clean",
    "D": PREPROC_ROOT / "Trial_D (Resize + Contrast Stretching + CLAHE + Median Filter)" / "CombinedPaper_Clean",
}
BASE_ROOT = PREPROC_TRIALS["A"]          # SesameBase = Trial A of CombinedPaper

OUT_ROOT = HERE / "outputs"
# Images to leave out of every experiment (uid,reason). Edit by hand if needed.
EXCLUDE_FILE = HERE / "exclude.csv"
META_DIR = OUT_ROOT / "metadata"
SPLIT_DIR = OUT_ROOT / "splits"
RUNS_DIR = OUT_ROOT / "runs"

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
CLASSES = ["Bacteria Blight Infected", "Healthy Leaf", "Phyllody Infected", "Powdery Mildew"]
CLASS_CODE = {"Bacteria Blight Infected": "BB", "Healthy Leaf": "HL",
              "Phyllody Infected": "PI", "Powdery Mildew": "PM"}
SYN_SUFFIX = " - Synthetic"
IMG_EXT = (".jpg", ".jpeg", ".png")

# Fixed seed for the splits. Never change this after the splits are made.
SPLIT_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5          # validation = 1/5 of each training part


# --------------------------------------------------------------------------
# File-name helpers
# --------------------------------------------------------------------------
def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def strip_trial_prefix(fname):
    """'Trial_A_00012.jpg' -> '00012.jpg' (works for Trial A/B/C/D)."""
    return re.sub(r"^Trial_[A-D]_", "", fname)


def make_uid(cls, image_id, synthetic=False):
    """Unique id. ERFD file names repeat across classes, so the class is part of it."""
    return f"{CLASS_CODE[cls]}{'-SYN' if synthetic else ''}/{image_id}"


def parse_source(image_id, synthetic=False):
    """Which original dataset an image came from, read from its file name."""
    if synthetic:
        return "GEMINI"
    if ".rf." in image_id:
        return "RSFD"            # Roboflow
    if re.search(r"_bg\d+", image_id):
        return "MSLD"            # Mendeley lab image composited on a field background
    if re.fullmatch(r"\d{5}\.(jpg|jpeg|png)", image_id, flags=re.I):
        return "ERFD"            # Hailu + Nibret field images
    return "UNKNOWN"


def parse_time(image_id, source):
    """Photo time from the file name, if the name has one (else None)."""
    if source == "MSLD":
        m = re.search(r"IMG(\d{14})", image_id)
        if m:
            return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    if source == "RSFD":
        m = re.match(r"(\d{8})_(\d{6})", image_id)
        if m:
            return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return None


def parse_camera_seq(image_id, source):
    """Camera counter for names like 'IMG_0375_jpg.rf...' (else None)."""
    if source == "RSFD":
        m = re.match(r"IMG_(\d+)", image_id)
        if m:
            return int(m.group(1))
    return None


def gemini_parent(image_id):
    """Synthetic tiles 'syn_0007_top_left.jpg' all come from Gemini image 'syn_0007'."""
    m = re.match(r"(syn_\d+)", image_id)
    return m.group(1) if m else Path(image_id).stem


# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def text_md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def environment_info():
    """Versions and hardware, saved into every output file."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "time": dt.datetime.now().isoformat(timespec="seconds"),
    }
    for mod in ["torch", "torchvision", "sklearn", "pandas", "scipy", "efficient_kan", "openpyxl"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            info[mod] = "not installed"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        pass
    return info


class Tee:
    """Print to the console AND append to a log file."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
