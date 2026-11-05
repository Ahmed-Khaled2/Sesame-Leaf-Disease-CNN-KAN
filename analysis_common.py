"""
analysis_common.py - small helpers shared by gradcam.py, kan_curves.py and bench_speed.py.
Not meant to be run on its own.
"""
import glob
import json
import os
import re
import shlex
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import train
from common import GP_ROOT, RUNS_DIR, OUT_ROOT, CLASS_CODE

FIG_DIR = OUT_ROOT / "figures"


def run_args(run_name):
    """Rebuild the train.py arguments of a finished run from its config.json."""
    run_dir = RUNS_DIR / run_name
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    argv = shlex.split(cfg["command"])
    argv = argv[argv.index(next(x for x in argv if x.endswith("train.py"))) + 1:]
    return train.get_args(argv), cfg, run_dir


def weights_path(run_dir, fold):
    hits = sorted(glob.glob(str(run_dir / "weights" / f"fold{fold}_best_epoch*.pth")))
    return Path(hits[0]) if hits else None


def load_model(a, n_classes, run_dir, fold, device):
    """The saved best model of one fold, or None if the weights were not saved."""
    w = weights_path(run_dir, fold)
    if w is None:
        return None
    model = train.HybridNet(a, n_classes)
    model.load_state_dict(torch.load(w, map_location="cpu"))
    return model.to(device).eval()


_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(train.MEAN, train.STD)])


def load_photo(file):
    """(PIL image 224x224, normalised tensor) exactly as the model saw it at test time."""
    img = Image.open(GP_ROOT / file).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.BICUBIC)
    return img, _to_tensor(img)


def test_fold_of(sp, uid):
    row = sp[sp.uid == uid].iloc[0]
    for c in sp.columns:
        if c.startswith("fold") and row[c] == "test":
            return int(c[4:])
    return None


def pick_photos(sp, per_group, seed=0):
    """The same photos for every model: `per_group` test photos per class + source."""
    real = sp[~sp.is_synthetic].sort_values("uid")
    rng = np.random.default_rng(seed)
    rows = []
    for (cls, src), g in sorted(real.groupby(["cls", "source"]), key=lambda t: t[0]):
        idx = rng.permutation(len(g))[:per_group]
        for i in sorted(idx):
            r = g.iloc[i]
            rows.append(dict(uid=r.uid, cls=cls, source=src, file=r.file, fold=test_fold_of(sp, r.uid)))
    return rows


def short(cls):
    return CLASS_CODE.get(cls, cls[:2])


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def wait_for_other_workers(max_hours=12, poll_s=60):
    """When started from run_queue.py, wait until every OTHER worker of the same queue has
    finished, so nothing else uses the GPU. Does nothing when run by hand."""
    name, me, n = os.environ.get("QUEUE_NAME"), os.environ.get("QUEUE_WORKER"), os.environ.get("QUEUE_WORKERS")
    if not (name and me is not None and n):
        return
    me, n = int(me), int(n)
    logs = OUT_ROOT / "queue_logs"
    start = time.time()
    while True:
        waiting = []
        for w in range(n):
            if w == me:
                continue
            p = logs / f"{name}_worker{w}.txt"
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            starts = [m.start() for m in re.finditer(rf"=== .* worker {w}/{n}:", text)]
            last = text[starts[-1]:] if starts else ""
            if not last or "finished" not in last:
                waiting.append(w)
        if not waiting:
            print("All other workers have finished.", flush=True)
            return
        if time.time() - start > max_hours * 3600:
            print(f"Still waiting for workers {waiting} after {max_hours} h - going ahead anyway.", flush=True)
            return
        print(f"{time.strftime('%H:%M:%S')}  waiting for workers {waiting} to finish ...", flush=True)
        time.sleep(poll_s)
