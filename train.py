"""
train.py  -  STEP 3 of the journal re-run: ONE run = one dataset + one head + one seed, all 5 folds.

Same pipeline as the GP (ResNet backbone, KAN or MLP head, same augmentations,
AdamW, same learning rates, batch 64, 50 epochs), with these fixes:
  1. All seeds are fixed (Python, NumPy, PyTorch, sampling, shuffling).
  2. Frozen backbone layers really stay frozen (their BatchNorm stays in eval mode).
  3. Balancing is redone EVERY epoch from ALL real training images. The target per
     class is (oversample x smallest REAL class); synthetic images never change it.
  4. Optional LayerNorm in front of every head layer (same for KAN, MLP and linear),
     so KAN inputs sit inside the spline grid.  --norm none = your original heads.
  5. Best epoch is chosen on the VALIDATION set. The test set is never used to choose
     anything - its scores are only logged for the curves.
  6. KAN checks after every fold: how much the splines contribute (kan_check), how much of
     what each class output receives comes from the splines (kan_class_share), and what
     happens to test accuracy when the spline part or the base part is switched off
     (kan_knockout).  --kan-base off trains a spline-only KAN (base term fixed at zero).
  7. --act silu gives the MLP head SiLU instead of ReLU (the MLP-SiLU baseline).
  8. --data multi_syn2 = SesameMulti + the second synthetic set (made without any real photo
     as a sample). Its images come in 5 batches and batch k is never used in fold k, so every
     fold trains on 4 batches (see make_syn2_split.py).

Everything is logged (see outputs/runs/<run name>/):
  run_log.xlsx   config | epochs | folds | per_class | confusion | predictions |
                 sampling | kan_check | kan_class_share | kan_knockout | model_info | summary
  *.csv          the same tables (epochs.csv is written after every epoch)
  curves.png     mean curves over the finished folds
  console_log.txt, STATUS.txt, weights/ (only with --save-weights best)

If a run stops (crash, power cut), start the same command again: finished folds are
kept and it continues from the next fold.

Examples:
  python train.py --data multi --head kan --seed 0
  python train.py --data multi --head mlp --hidden 1536 --seed 0        (size-matched MLP)
  python train.py --data multi --head mlp --act silu --seed 0           (MLP-SiLU)
  python train.py --data multi_syn --head kan --seed 1
  python train.py --data multi_syn2 --head kan --seed 1                 (second synthetic set)
  python train.py --data base --preproc C --head mlp --unfreeze none --seed 0
  python train.py --data multi --head kan --norm none --seed 0          (original KAN)
  python train.py --data multi --head kan --kan-base off --seed 0       (spline-only KAN)
  python train.py --data multi --backbone efficientnet_b0 --head kan --seed 0
  python train.py --data multi --train-frac 0.25 --head mlp --seed 0     (25% of the training photos)
  python train.py --data base --split random --head kan --norm none --seed 0   (old-style random split)
  python train.py --data bbhl_cross --head kan --oversample 1 --seed 0  (train on one source, test on the other)
"""
import argparse
import copy
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (GP_ROOT, SPLIT_DIR, META_DIR, RUNS_DIR, PREPROC_TRIALS, CLASSES, N_OUTER_FOLDS,
                    set_all_seeds, file_md5, text_md5, environment_info, Tee)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (f1_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, recall_score)

try:
    from efficient_kan import KAN, KANLinear
except ImportError:
    from efficient_kan.kan import KAN, KANLinear

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BACKBONES = {"resnet18": models.resnet18, "resnet34": models.resnet34,
             "resnet50": models.resnet50, "resnet101": models.resnet101,
             "efficientnet_b0": models.efficientnet_b0}
BACKBONE_SHORT = {"resnet18": "r18", "resnet34": "r34", "resnet50": "r50", "resnet101": "r101",
                  "efficientnet_b0": "effb0"}
FEATURE_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "efficientnet_b0": 1280}
# what "--unfreeze layer4" trains for each family (the last part of the backbone)
UNFROZEN_PART = {"resnet": "layer4", "efficientnet": "features[6], features[7], features[8] (last two stages + final conv)"}
DATA_CHOICES = ["base", "multi", "multi_syn", "multi_syn2", "bbhl_cross", "bbhl_mixed"]
SYN_DATA = ("multi_syn", "multi_syn2")       # the datasets that keep their synthetic rows


# ==========================================================================
# Arguments and run name
# ==========================================================================
def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, choices=DATA_CHOICES,
                    help="bbhl_cross = Bacterial Blight vs Healthy, trained on one source (ERFD or MSLD) and "
                         "tested on the other; bbhl_mixed = the same two classes and sources with normal splits")
    ap.add_argument("--split", default="group", choices=["group", "random"],
                    help="random = photo-level splits that ignore near-duplicate groups (only for a leakage check)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="use only this share of the training groups (e.g. 0.25); val and test stay the same")
    ap.add_argument("--preproc", default="A", choices=["A", "B", "C", "D"], help="preprocessing trial (base only)")
    ap.add_argument("--head", required=True, choices=["kan", "mlp", "linear"])
    ap.add_argument("--hidden", default="128", help="hidden widths, e.g. 128 or 128,64")
    ap.add_argument("--grid", type=int, default=5, help="KAN grid size")
    ap.add_argument("--order", type=int, default=5, help="KAN spline order")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--norm", default="ln", choices=["ln", "none"], help="LayerNorm before each head layer")
    ap.add_argument("--kan-base", default="on", choices=["on", "off"],
                    help="KAN base term (SiLU + linear). off = spline-only KAN")
    ap.add_argument("--act", default="relu", choices=["relu", "silu"],
                    help="activation of the MLP head. silu = the MLP-SiLU baseline")
    ap.add_argument("--backbone", default="resnet50", choices=list(BACKBONES))
    ap.add_argument("--unfreeze", default="layer4", choices=["layer4", "none"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW default, same as the GP")
    ap.add_argument("--oversample", type=float, default=4.0, help="target per class = this x smallest real class")
    ap.add_argument("--select", default="val_f1", choices=["val_f1", "val_acc", "val_loss"],
                    help="how the best epoch is chosen (validation only)")
    ap.add_argument("--folds", default="", help="which outer folds to run (default: all folds in the split file)")
    ap.add_argument("--save-weights", default="none", choices=["none", "best"])
    ap.add_argument("--phase", default="P0", help="label for the experiment phase, e.g. P1, P2, P3")
    ap.add_argument("--name", default="", help="optional extra text for the run name")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="tiny test run (2 epochs, few batches)")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run with the same name")
    a = ap.parse_args(argv)
    a.hidden_list = [int(h) for h in a.hidden.split(",") if h.strip()] if a.head != "linear" else []
    a.fold_list = [int(f) for f in a.folds.split(",") if f.strip()]
    if a.data != "base" and a.preproc != "A":
        ap.error("--preproc B/C/D only exists for --data base")
    if a.kan_base == "off" and a.head != "kan":
        ap.error("--kan-base off only makes sense with --head kan")
    if a.act != "relu" and a.head != "mlp":
        ap.error("--act only applies to --head mlp")
    if a.split == "random" and a.data not in ("base", "multi", "multi_syn"):
        ap.error("--split random exists only for base / multi / multi_syn")
    if not 0 < a.train_frac <= 1:
        ap.error("--train-frac must be between 0 and 1")
    if a.smoke:
        a.epochs = 2
    return a


def run_name(a):
    data = a.data + (f"-{a.preproc}" if a.data == "base" else "")
    if a.split == "random":
        data += "-rand"
    if a.train_frac < 1:
        data += f"-f{round(a.train_frac * 100)}"
    parts = [a.phase, data, a.head]
    if a.head != "linear":
        parts.append("h" + "x".join(map(str, a.hidden_list)))
    if a.head == "mlp" and a.act != "relu":
        parts.append(a.act)                  # ReLU runs keep their old names
    if a.head == "kan":
        parts += [f"g{a.grid}", f"o{a.order}"] + (["nobase"] if a.kan_base == "off" else [])
    parts += [f"d{a.dropout}", a.norm, "l4" if a.unfreeze == "layer4" else "frozen",
              BACKBONE_SHORT[a.backbone], f"s{a.seed}"]
    if a.name:
        parts.append(a.name)
    if a.smoke:
        parts.append("SMOKE")
    return "_".join(parts)


# ==========================================================================
# Data
# ==========================================================================
def split_file(a):
    if a.data in ("bbhl_cross", "bbhl_mixed", "multi_syn2"):
        return SPLIT_DIR / f"splits_{a.data}.csv"
    name = "splits_base" if a.data == "base" else "splits_multi"
    return SPLIT_DIR / (name + ("_random" if a.split == "random" else "") + ".csv")


def load_split(a):
    path = split_file(a)
    if not path.exists():
        raise SystemExit(f"{path} not found - run make_splits.py (and make_extra_splits.py) first.")
    sp = pd.read_csv(path, keep_default_na=False)
    sp["is_synthetic"] = sp.is_synthetic.astype(str).str.lower().isin(["true", "1"])
    if a.data not in SYN_DATA:
        sp = sp[~sp.is_synthetic].reset_index(drop=True)
    if a.data == "base" and a.preproc != "A":
        root = PREPROC_TRIALS[a.preproc]
        sp["file"] = [str((root / c / f"Trial_{a.preproc}_{u.split('/', 1)[1]}").relative_to(GP_ROOT))
                      for c, u in zip(sp.cls, sp.uid)]
    classes = [c for c in CLASSES if c in set(sp.cls)]
    sp["label"] = sp.cls.map({c: i for i, c in enumerate(classes)})
    raw = pd.read_csv(path, keep_default_na=False)          # same fingerprint as make_splits.py prints
    fold_cols = [c for c in raw.columns if c.startswith("fold")]
    fp = text_md5(raw.sort_values("uid")[["uid"] + fold_cols].to_csv(index=False, lineterminator="\n"))
    folds = sorted(int(c[4:]) for c in fold_cols)
    return sp, classes, path, fp, folds


class ImageCache:
    """Loads every image once. Training uses PIL images (augmented on the fly, like the GP);
    validation/test use ready tensors."""
    def __init__(self, files, log):
        self.pil, self.tensor = {}, {}
        to_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        resized = 0
        for f in files:
            img = Image.open(GP_ROOT / f).convert("RGB")
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BICUBIC)
                resized += 1
            img.load()
            self.pil[f] = img
            self.tensor[f] = to_t(img)
        log(f"  images loaded into RAM: {len(files)} (resized to 224: {resized})")


class TrainSet(Dataset):
    def __init__(self, cache, files, labels):
        self.cache, self.files, self.labels = cache, files, labels
        self.aug = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.aug(self.cache.pil[self.files[i]]), self.labels[i]


class EvalSet(Dataset):
    def __init__(self, cache, files, labels):
        self.cache, self.files, self.labels = cache, files, labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.cache.tensor[self.files[i]], self.labels[i]


def subsample_groups(train_df, frac, seed, fold, log):
    """Keep only `frac` of the REAL training groups, per class+source, so every class keeps the
    same share. The order of the groups depends only on (seed, fold), not on the head, so KAN and
    MLP get exactly the same photos, and the 25% set is inside the 50% set."""
    rng = np.random.default_rng(10_000 + 100 * seed + fold)
    real = train_df[~train_df.is_synthetic]
    keep = []
    for (cls, src), g in sorted(real.groupby(["cls", "source"]), key=lambda t: t[0]):
        groups = np.array(sorted(g.group_id.unique()))
        order = groups[rng.permutation(len(groups))]
        keep += list(order[:max(1, int(round(frac * len(groups))))])
    mask = train_df.is_synthetic | train_df.group_id.isin(set(keep))
    out = train_df[mask]
    log(f"  train-frac {frac}: kept {int((~out.is_synthetic).sum())} of {len(real)} real training photos "
        f"({len(keep)} of {real.group_id.nunique()} groups)")
    return out


def draw_epoch(train_df, target, rng):
    """Balanced draw for one epoch. Every image of a small class is used at least once,
    a big class is sub-sampled (a NEW subset every epoch, so all images get seen)."""
    rows, stats = [], []
    for label, g in train_df.groupby("label"):
        idx = g.index.to_numpy()
        if len(idx) >= target:
            chosen = rng.choice(idx, target, replace=False)
        else:
            extra = rng.choice(idx, target - len(idx), replace=True)
            chosen = np.concatenate([idx, extra])
        rows.append(chosen)
        sub = train_df.loc[chosen]
        stats.append(dict(label=int(label), cls=g.cls.iloc[0], target=int(target),
                          pool_real=int((~g.is_synthetic).sum()), pool_synthetic=int(g.is_synthetic.sum()),
                          drawn_total=len(chosen), drawn_real=int((~sub.is_synthetic).sum()),
                          drawn_synthetic=int(sub.is_synthetic.sum()),
                          unique_real=int(sub[~sub.is_synthetic].uid.nunique()),
                          unique_synthetic=int(sub[sub.is_synthetic].uid.nunique())))
    order = np.concatenate(rows)
    rng.shuffle(order)
    return order, stats


# ==========================================================================
# Model
# ==========================================================================
def load_backbone(name, tries=6):
    """ImageNet weights. Retries a few times, in case two workers download the same file at once."""
    for t in range(tries):
        try:
            return BACKBONES[name](weights="IMAGENET1K_V1")
        except Exception as e:                       # noqa: BLE001
            if t == tries - 1:
                raise
            print(f"  loading {name} weights failed ({type(e).__name__}: {e}); retrying in 20 s", flush=True)
            time.sleep(20)


class HybridNet(nn.Module):
    def __init__(self, a, n_classes):
        super().__init__()
        self.backbone = load_backbone(a.backbone)
        if a.backbone.startswith("resnet"):
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            last_part = [self.backbone.layer4]
        else:                                       # EfficientNet: features -> avgpool -> classifier
            feat_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
            last_part = [self.backbone.features[6], self.backbone.features[7], self.backbone.features[8]]
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.trainable_backbone = []
        if a.unfreeze == "layer4":
            for m in last_part:
                for p in m.parameters():
                    p.requires_grad = True
            self.trainable_backbone = last_part
        self.is_resnet = a.backbone.startswith("resnet")
        self.head_type, self.norm = a.head, a.norm
        # MLP hidden activation (runs made before --act existed have no "act" -> ReLU, as they were trained)
        self.act = F.silu if getattr(a, "act", "relu") == "silu" else F.relu
        self.dropout = nn.Dropout(a.dropout)
        dims = [feat_dim] + a.hidden_list + [n_classes]

        if a.head == "kan" and a.norm == "none":
            # exactly the GP head: efficient_kan KAN, no normalisation
            self.kan = KAN(dims, grid_size=a.grid, spline_order=a.order)
            self.kan_layers = list(self.kan.layers)
        elif a.head == "kan":
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[:-1]])
            self.kan_layers_m = nn.ModuleList([KANLinear(i, o, grid_size=a.grid, spline_order=a.order)
                                               for i, o in zip(dims[:-1], dims[1:])])
            self.kan_layers = list(self.kan_layers_m)
        else:
            # MLP (GP head: Linear -> ReLU -> Dropout -> ... -> Linear) or linear probe
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[:-1]]) if a.norm == "ln" else None
            self.linears = nn.ModuleList([nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])])
            self.kan_layers = []
        if a.head == "kan" and a.kan_base == "off":
            # spline-only KAN: the base term (SiLU -> linear) is fixed at zero and never trained
            for layer in self.kan_layers:
                layer.base_weight.data.zero_()
                layer.base_weight.requires_grad = False

    @property
    def cam_layer(self):
        """Last conv block, for Grad-CAM (a property, so it is not stored twice in the saved weights)."""
        return self.backbone.layer4 if self.is_resnet else self.backbone.features[8]

    def set_train_mode(self):
        """Head + unfrozen blocks train; frozen blocks (incl. their BatchNorm) stay in eval."""
        self.train()
        self.backbone.eval()
        for m in self.trainable_backbone:
            m.train()

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def features(self, x):
        return self.backbone(x)

    def head(self, f):
        """GP heads when --norm none:
             KAN: Dropout -> KAN(efficient_kan)
             MLP: Dropout -> Linear -> ReLU -> Dropout -> Linear   (--act silu: SiLU instead of ReLU)
           With --norm ln a LayerNorm goes in front of every head layer (the first one before the Dropout)."""
        if self.head_type == "kan" and self.norm == "none":
            return self.kan(self.dropout(f))
        x = f
        if self.head_type == "kan":
            for i, (ln, layer) in enumerate(zip(self.norms, self.kan_layers_m)):
                x = ln(x)
                if i == 0:
                    x = self.dropout(x)
                x = layer(x)
            return x
        n = len(self.linears)
        for i, lin in enumerate(self.linears):
            if self.norms is not None:
                x = self.norms[i](x)
            if i == 0:
                x = self.dropout(x)
            x = lin(x)
            if i < n - 1:
                x = self.dropout(self.act(x))
        return x

    def forward(self, x):
        return self.head(self.features(x))


def describe_head(a, n_classes):
    dims = [FEATURE_DIM[a.backbone]] + a.hidden_list + [n_classes]
    if a.head == "kan":
        steps = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            if a.norm == "ln":
                steps.append(f"LayerNorm({d_in})")
            if i == 0:
                steps.append(f"Dropout({a.dropout})")
            steps.append(f"KANLinear({d_in}->{d_out}, grid={a.grid}, order={a.order}"
                         + (", spline only)" if a.kan_base == "off" else ")"))
        return " -> ".join(steps)
    steps = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        if a.norm == "ln":
            steps.append(f"LayerNorm({d_in})")
        if i == 0:
            steps.append(f"Dropout({a.dropout})")
        steps.append(f"Linear({d_in}->{d_out})")
        if i < len(dims) - 2:
            steps += ["SiLU" if a.act == "silu" else "ReLU", f"Dropout({a.dropout})"]
    return " -> ".join(steps)


# ==========================================================================
# Evaluation helpers
# ==========================================================================
@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        probs.append(F.softmax(out, 1).cpu().numpy())
        labels.append(y.cpu().numpy())
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    pred = p.argmax(1)
    rec = recall_score(y, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    return dict(loss=loss_sum / len(y), acc=100 * float((pred == y).mean()),
                f1=float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
                bal_acc=100 * float(balanced_accuracy_score(y, pred)),
                recall=rec, probs=p, y=y, pred=pred)


@torch.no_grad()
def kan_check(model, loader, device, fold):
    """How much the splines actually do, measured on the test images with the best weights."""
    rows = []
    model.eval()
    feats = []
    captured = {}
    hooks = []
    for li, layer in enumerate(model.kan_layers):
        hooks.append(layer.register_forward_pre_hook(
            lambda m, inp, li=li: captured.setdefault(li, []).append(inp[0].detach().float().cpu())))
    for x, _ in loader:
        x = x.to(device)
        f = model.features(x)
        feats.append(f.float().cpu())
        model.head(f)
    for h in hooks:
        h.remove()
    f = torch.cat(feats)
    rows.append(dict(fold=fold, layer="backbone output (head input before norm)", n_values=f.numel(),
                     mean=f.mean().item(), std=f.std().item(), min=f.min().item(), max=f.max().item(),
                     share_in_minus1_1=((f >= -1) & (f <= 1)).float().mean().item()))
    for li, layer in enumerate(model.kan_layers):
        x = torch.cat(captured[li]).to(device)
        g = layer.grid
        k = layer.spline_order
        core_lo, core_hi = g[:, k], g[:, -k - 1]
        sup_lo, sup_hi = g[:, 0], g[:, -1]
        base = F.linear(layer.base_activation(x), layer.base_weight)
        spline = F.linear(layer.b_splines(x).view(x.size(0), -1),
                          layer.scaled_spline_weight.view(layer.out_features, -1))
        init = KANLinear(layer.in_features, layer.out_features, grid_size=layer.grid_size,
                         spline_order=k).grid.to(g.device)
        rows.append(dict(
            fold=fold, layer=f"KAN layer {li + 1} input", n_values=x.numel(),
            mean=x.mean().item(), std=x.std().item(), min=x.min().item(), max=x.max().item(),
            share_in_minus1_1=((x >= -1) & (x <= 1)).float().mean().item(),
            share_in_grid_core=((x >= core_lo) & (x <= core_hi)).float().mean().item(),
            share_in_spline_support=((x > sup_lo) & (x < sup_hi)).float().mean().item(),
            mean_abs_base_term=base.abs().mean().item(), mean_abs_spline_term=spline.abs().mean().item(),
            spline_share_of_output=(spline.abs().mean() / (spline.abs().mean() + base.abs().mean())).item(),
            grid_unchanged_from_init=bool(torch.allclose(g, init)),
            grid_first=g[0, 0].item(), grid_last=g[0, -1].item()))
    return rows


@torch.no_grad()
def kan_class_share(model, loader, device, fold, classes):
    """Per class: how much of what the class output receives comes from the splines.
    Last KAN layer, test images, best weights. For every connection i -> class j:
    mean |spline_ji(x_i)| and mean |w_ji * SiLU(x_i)| over the test images; the share is
    sum_i spline / (sum_i spline + sum_i base). Same definition as kan_curves.py."""
    layer = model.kan_layers[-1]
    model.eval()
    store = []
    hook = layer.register_forward_pre_hook(lambda m, inp: store.append(inp[0].detach().float()))
    for x, _ in loader:
        model(x.to(device))
    hook.remove()
    X = torch.cat(store)
    B = layer.b_splines(X)                                   # (N, in, coeff)
    S = layer.scaled_spline_weight                           # (out, in, coeff)
    act = layer.base_activation(X)                           # (N, in)
    rows, tot_s, tot_b = [], 0.0, 0.0
    for j, c in enumerate(classes):
        s = torch.einsum("nic,ic->ni", B, S[j]).abs().mean(0).sum().item()
        b = (act * layer.base_weight[j]).abs().mean(0).sum().item()
        tot_s, tot_b = tot_s + s, tot_b + b
        rows.append(dict(fold=fold, cls=c, n_test_images=X.size(0), sum_mean_abs_spline=s, sum_mean_abs_base=b,
                         spline_share=s / max(s + b, 1e-12)))
    rows.append(dict(fold=fold, cls="all classes", n_test_images=X.size(0), sum_mean_abs_spline=tot_s,
                     sum_mean_abs_base=tot_b, spline_share=tot_s / max(tot_s + tot_b, 1e-12)))
    return rows


@torch.no_grad()
def kan_knockout(model, loader, device, n_classes, fold):
    """Test scores of the trained KAN when one part of every KAN layer is switched off.
    If removing the splines changes nothing, the KAN works like a plain SiLU network."""
    full = evaluate(model, loader, device, n_classes)
    rows = [dict(fold=fold, variant="full KAN (as trained)", test_acc=full["acc"], test_f1=full["f1"],
                 change_in_acc=0.0, change_in_f1=0.0)]
    variants = [("spline part removed (all KAN layers)", "spline", None),
                ("spline part removed (KAN layer 1 only)", "spline", 0),
                ("spline part removed (last KAN layer only)", "spline", len(model.kan_layers) - 1),
                ("base part removed (all KAN layers)", "base", None)]
    for label, part, only in variants:
        saved = []
        for li, layer in enumerate(model.kan_layers):
            if only is not None and li != only:
                continue
            w = layer.spline_weight if part == "spline" else layer.base_weight
            saved.append((w, w.data.clone()))
            w.data.zero_()
        ev = evaluate(model, loader, device, n_classes)
        for w, d in saved:
            w.data.copy_(d)
        rows.append(dict(fold=fold, variant=label, test_acc=ev["acc"], test_f1=ev["f1"],
                         change_in_acc=ev["acc"] - full["acc"], change_in_f1=ev["f1"] - full["f1"]))
    return rows


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.head_params())
    return total, trainable, head


def macs_info(model, device):
    try:
        from thop import profile
        m = copy.deepcopy(model).eval()
        macs, _ = profile(m, inputs=(torch.randn(1, 3, 224, 224, device=device),), verbose=False)
        del m
        return float(macs)
    except Exception as e:     # thop missing, or cannot count a layer
        return f"not measured ({type(e).__name__})"


# ==========================================================================
# Logging to CSV / Excel
# ==========================================================================
class RunLogger:
    TABLES = ["epochs", "folds", "per_class", "confusion", "predictions", "sampling", "kan_check",
              "kan_class_share", "kan_knockout"]

    def __init__(self, run_dir):
        self.dir = run_dir
        self.tables = {t: [] for t in self.TABLES}
        for t in self.TABLES:
            p = run_dir / f"{t}.csv"
            if p.exists() and p.stat().st_size > 5:
                try:
                    self.tables[t] = pd.read_csv(p, keep_default_na=False).to_dict("records")
                except pd.errors.EmptyDataError:
                    pass

    def add(self, table, rows):
        rows = rows if isinstance(rows, list) else [rows]
        self.tables[table].extend(rows)

    def drop_fold(self, fold):
        for t in self.TABLES:
            self.tables[t] = [r for r in self.tables[t] if str(r.get("fold")) != str(fold)]

    def flush_csv(self, table):
        pd.DataFrame(self.tables[table]).to_csv(self.dir / f"{table}.csv", index=False, encoding="utf-8")

    def flush_all(self, config, model_info):
        for t in self.TABLES:
            self.flush_csv(t)
        summary = make_summary(pd.DataFrame(self.tables["folds"]))
        summary.to_csv(self.dir / "summary.csv", index=False, encoding="utf-8")
        tmp = self.dir / "run_log.tmp.xlsx"
        with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
            pd.DataFrame([dict(setting=k, value=str(v)) for k, v in config.items()]).to_excel(
                xw, sheet_name="config", index=False)
            summary.to_excel(xw, sheet_name="summary", index=False)
            for t in self.TABLES:
                pd.DataFrame(self.tables[t]).to_excel(xw, sheet_name=t, index=False)
            pd.DataFrame([dict(item=k, value=str(v)) for k, v in model_info.items()]).to_excel(
                xw, sheet_name="model_info", index=False)
        os.replace(tmp, self.dir / "run_log.xlsx")


def make_summary(folds):
    if folds.empty:
        return pd.DataFrame()
    cols = ["test_acc", "test_f1", "test_bal_acc", "test_loss", "val_acc", "val_f1", "val_loss",
            "best_epoch", "test_acc_last_epoch", "test_f1_last_epoch", "test_acc_mean_last5",
            "test_f1_mean_last5", "train_acc_at_best", "fold_minutes"]
    rows = []
    for c in cols:
        if c not in folds:
            continue
        v = pd.to_numeric(folds[c], errors="coerce").dropna()
        if v.empty:
            continue
        rows.append(dict(metric=c, n_folds=len(v), mean=v.mean(), std_sample=v.std(ddof=1) if len(v) > 1 else 0.0,
                         min=v.min(), max=v.max()))
    return pd.DataFrame(rows)


def save_curves(run_dir, epochs_df, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if epochs_df.empty:
        return
    e = epochs_df.copy()
    for c in e.columns:
        if c not in ("fold",):
            e[c] = pd.to_numeric(e[c], errors="coerce")
    m = e.groupby("epoch").mean(numeric_only=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for split, style in (("train", "-"), ("val", "--"), ("test", ":")):
        ax[0].plot(m.index, m[f"{split}_loss"], style, label=split)
        ax[1].plot(m.index, m[f"{split}_acc"], style, label=split)
        ax[2].plot(m.index, m[f"{split}_f1"], style, label=split)
    for a_, t in zip(ax, ("loss", "accuracy (%)", "macro-F1")):
        a_.set_title(t); a_.set_xlabel("epoch"); a_.legend(); a_.grid(alpha=0.3)
    fig.suptitle(f"{title}  (mean of {e.fold.nunique()} folds; test is logged only, never used to choose)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=90)
    plt.close(fig)


# ==========================================================================
# One fold
# ==========================================================================
def run_fold(a, k, sp, classes, cache, device, logger, log, run_dir):
    n_classes = len(classes)
    fold_seed = a.seed * 1000 + k
    set_all_seeds(fold_seed)
    rng = np.random.default_rng(fold_seed)
    col = f"fold{k}"

    train_df = sp[sp[col] == "train"]
    val_df = sp[sp[col] == "val"]
    test_df = sp[sp[col] == "test"]
    if a.train_frac < 1:
        train_df = subsample_groups(train_df, a.train_frac, a.seed, k, log)
    real_counts = train_df[~train_df.is_synthetic].label.value_counts()
    target = int(round(a.oversample * real_counts.min()))
    log(f"\n--- fold {k} | train real {int((~train_df.is_synthetic).sum())} + synthetic "
        f"{int(train_df.is_synthetic.sum())} | val {len(val_df)} | test {len(test_df)} | "
        f"target per class {target} (= {a.oversample} x {real_counts.min()}) | seed {fold_seed}")

    val_loader = DataLoader(EvalSet(cache, val_df.file.tolist(), val_df.label.tolist()), batch_size=a.batch)
    test_loader = DataLoader(EvalSet(cache, test_df.file.tolist(), test_df.label.tolist()), batch_size=a.batch)

    model = HybridNet(a, n_classes).to(device)
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": model.head_params(), "lr": a.lr_head}]
    if bb_params:
        groups.insert(0, {"params": bb_params, "lr": a.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)

    best, best_key, best_state = None, None, None
    best_val_eval, best_test_eval = None, None
    t_fold = time.time()
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        order, stats = draw_epoch(train_df, target, rng)
        for s in stats:
            logger.add("sampling", dict(fold=k, epoch=epoch, **s))
        ep_df = train_df.loc[order]
        g = torch.Generator().manual_seed(fold_seed * 100 + epoch)
        loader = DataLoader(TrainSet(cache, ep_df.file.tolist(), ep_df.label.tolist()),
                            batch_size=a.batch, shuffle=True, generator=g, num_workers=0)
        model.set_train_mode()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        tl, tc, tn, tp, ty = 0.0, 0, 0, [], []
        for b, (x, y) in enumerate(loader):
            if a.smoke and b >= 2:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            pred = out.argmax(1)
            tc += (pred == y).sum().item()
            tn += len(y)
            tp.append(pred.cpu().numpy()); ty.append(y.cpu().numpy())
        tp, ty = np.concatenate(tp), np.concatenate(ty)
        train_time = time.time() - t0

        ve = evaluate(model, val_loader, device, n_classes)
        te = evaluate(model, test_loader, device, n_classes)
        key = {"val_f1": (ve["f1"], -ve["loss"]), "val_acc": (ve["acc"], -ve["loss"]),
               "val_loss": (-ve["loss"], ve["f1"])}[a.select]
        is_best = best_key is None or key > best_key
        if is_best:
            best_key, best = key, epoch
            best_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
            best_val_eval, best_test_eval = ve, te
            best_train = dict(acc=100 * tc / tn, loss=tl / tn)

        row = dict(fold=k, epoch=epoch,
                   train_loss=tl / tn, train_acc=100 * tc / tn,
                   train_f1=f1_score(ty, tp, average="macro", zero_division=0),
                   val_loss=ve["loss"], val_acc=ve["acc"], val_f1=ve["f1"], val_bal_acc=ve["bal_acc"],
                   test_loss=te["loss"], test_acc=te["acc"], test_f1=te["f1"], test_bal_acc=te["bal_acc"],
                   is_best_so_far=is_best, train_images=tn,
                   train_seconds=round(train_time, 2), epoch_seconds=round(time.time() - t0, 2),
                   gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.startswith("cuda") else 0)
        for i, c in enumerate(classes):
            row[f"val_recall_{c}"] = float(ve["recall"][i])
            row[f"test_recall_{c}"] = float(te["recall"][i])
        logger.add("epochs", row)
        logger.flush_csv("epochs")
        logger.flush_csv("sampling")
        (run_dir / "STATUS.txt").write_text(f"running fold {k} epoch {epoch}/{a.epochs}  ({dt.datetime.now():%H:%M:%S})")
        if epoch == 1 or epoch % 5 == 0 or epoch == a.epochs:
            log(f"  ep {epoch:3d} | train loss {tl / tn:.4f} acc {100 * tc / tn:6.2f} | val acc {ve['acc']:6.2f} "
                f"F1 {ve['f1']:.4f} | test acc {te['acc']:6.2f} | best ep {best} | {time.time() - t0:.1f}s")

    # ---------- fold results at the best (validation-chosen) epoch
    last = [r for r in logger.tables["epochs"] if str(r["fold"]) == str(k)]
    last5 = last[-5:]
    p, r_, f, s = precision_recall_fscore_support(best_test_eval["y"], best_test_eval["pred"],
                                                  labels=list(range(n_classes)), zero_division=0)
    logger.add("folds", dict(
        fold=k, best_epoch=best, selected_by=a.select,
        test_acc=best_test_eval["acc"], test_f1=best_test_eval["f1"], test_bal_acc=best_test_eval["bal_acc"],
        test_loss=best_test_eval["loss"],
        val_acc=best_val_eval["acc"], val_f1=best_val_eval["f1"], val_loss=best_val_eval["loss"],
        train_acc_at_best=best_train["acc"], train_loss_at_best=best_train["loss"],
        test_acc_last_epoch=last[-1]["test_acc"], test_f1_last_epoch=last[-1]["test_f1"],
        test_acc_mean_last5=float(np.mean([r["test_acc"] for r in last5])),
        test_f1_mean_last5=float(np.mean([r["test_f1"] for r in last5])),
        n_train_real=int((~train_df.is_synthetic).sum()), n_train_synthetic=int(train_df.is_synthetic.sum()),
        n_val=len(val_df), n_test=len(test_df), target_per_class=target,
        fold_minutes=round((time.time() - t_fold) / 60, 2),
        **{f"n_test_{c}": int((test_df.cls == c).sum()) for c in classes},
        **{f"n_test_src_{s_}": int((test_df.source == s_).sum()) for s_ in sorted(sp.source.unique()) if s_ != "GEMINI"}))
    for i, c in enumerate(classes):
        logger.add("per_class", dict(fold=k, split="test", cls=c, precision=p[i], recall=r_[i], f1=f[i], support=int(s[i])))
    pv, rv, fv, sv = precision_recall_fscore_support(best_val_eval["y"], best_val_eval["pred"],
                                                     labels=list(range(n_classes)), zero_division=0)
    for i, c in enumerate(classes):
        logger.add("per_class", dict(fold=k, split="val", cls=c, precision=pv[i], recall=rv[i], f1=fv[i], support=int(sv[i])))
    cm = confusion_matrix(best_test_eval["y"], best_test_eval["pred"], labels=list(range(n_classes)))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            logger.add("confusion", dict(fold=k, split="test", true=ci, pred=cj, count=int(cm[i, j])))

    def pred_rows(df, ev, split, which):
        out = []
        for (_, r), pr, pd_ in zip(df.iterrows(), ev["probs"], ev["pred"]):
            row = dict(fold=k, split=split, epoch=which, uid=r.uid, file=r.file, cls=r.cls, source=r.source,
                       group_id=r.group_id, true=r.cls, predicted=classes[pd_], correct=bool(classes[pd_] == r.cls))
            for i, c in enumerate(classes):
                row[f"prob_{c}"] = round(float(pr[i]), 6)
            out.append(row)
        return out
    logger.add("predictions", pred_rows(test_df, best_test_eval, "test", "best"))
    logger.add("predictions", pred_rows(val_df, best_val_eval, "val", "best"))
    last_test = evaluate(model, test_loader, device, n_classes)          # model is at the last epoch now
    logger.add("predictions", pred_rows(test_df, last_test, "test", "last"))

    # ---------- spline check + weights, with the best weights loaded
    model.load_state_dict(best_state)
    if model.kan_layers:
        logger.add("kan_check", kan_check(model, test_loader, device, k))
        logger.add("kan_class_share", kan_class_share(model, test_loader, device, k, classes))
        logger.add("kan_knockout", kan_knockout(model, test_loader, device, n_classes, k))
    if a.save_weights == "best":
        wdir = run_dir / "weights"
        wdir.mkdir(exist_ok=True)
        torch.save(best_state, wdir / f"fold{k}_best_epoch{best}.pth")
    log(f"  fold {k} done: best epoch {best} | test acc {best_test_eval['acc']:.2f} F1 {best_test_eval['f1']:.4f} "
        f"| last-epoch test acc {last[-1]['test_acc']:.2f} | {(time.time() - t_fold) / 60:.1f} min")
    return model


# ==========================================================================
# main
# ==========================================================================
def main(argv=None):
    a = get_args(argv)
    name = run_name(a)
    run_dir = RUNS_DIR / name
    if run_dir.exists() and a.overwrite:
        import shutil
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(run_dir / "console_log.txt")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    sp, classes, split_path, split_fp, split_folds = load_split(a)
    if not a.fold_list:
        a.fold_list = split_folds
    bad = [f for f in a.fold_list if f not in split_folds]
    if bad:
        raise SystemExit(f"folds {bad} do not exist in {split_path.name} (it has folds {split_folds})")
    meta_fp = text_md5((META_DIR / "metadata.csv").read_text(encoding="utf-8")) if (META_DIR / "metadata.csv").exists() else ""
    here = Path(__file__).resolve().parent
    config = dict(
        run_name=name, **{k: v for k, v in vars(a).items() if k not in ("hidden_list", "fold_list")},
        hidden_list=a.hidden_list, classes=classes, n_classes=len(classes),
        head_structure=describe_head(a, len(classes)),
        frozen_part=(f"all backbone layers except {UNFROZEN_PART['resnet' if a.backbone.startswith('resnet') else 'efficientnet']}"
                     if a.unfreeze == "layer4" else "whole backbone"),
        split_folds=split_folds,
        frozen_batchnorm="eval mode (running stats not updated)",
        augmentation="RandomRotation(30), RandomHorizontalFlip, ColorJitter(0.1,0.1,0.05,0.02), ImageNet normalisation",
        balancing=f"every epoch: target = {a.oversample} x smallest REAL train class; small classes: all images once + random extra; big classes: new random subset each epoch",
        loss="cross-entropy", scheduler="none", optimizer="AdamW",
        split_file=str(split_path), split_fingerprint=split_fp, metadata_fingerprint=meta_fp,
        code_fingerprint_train=file_md5(here / "train.py"), code_fingerprint_common=file_md5(here / "common.py"),
        command="python " + " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        **{"env." + k: v for k, v in environment_info().items()})
    config["device_used"] = device

    cfg_path = run_dir / "config.json"
    compare_keys = [k for k in config if not k.startswith("env.") and k not in ("command", "device", "device_used", "folds", "save_weights", "overwrite")]
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        diff = [k for k in compare_keys if str(old.get(k)) != str(config.get(k))]
        if diff:
            raise SystemExit(f"A run named {name} already exists with different settings: {diff}\n"
                             f"Use --overwrite to replace it, or --name to give this run another name.")
    cfg_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    logger = RunLogger(run_dir)
    done = sorted({int(r["fold"]) for r in logger.tables["folds"]})
    todo = [f for f in a.fold_list if f not in done]
    log("=" * 80)
    log(f"RUN {name}  started {dt.datetime.now():%Y-%m-%d %H:%M:%S} on {device}")
    log(f"  head: {config['head_structure']}")
    log(f"  folds done earlier: {done}  -> to run now: {todo}")
    if not todo:
        log("  nothing to do - this run is complete.")
        return run_dir
    for f in todo:                       # remove half-finished rows of a fold that crashed
        logger.drop_fold(f)

    cache = ImageCache(sp.file.tolist(), log)
    model_info = {}
    for k in todo:
        model = run_fold(a, k, sp, classes, cache, device, logger, log, run_dir)
        if not model_info:
            total, trainable, head = count_params(model)
            head_trainable = sum(p.numel() for p in model.head_params() if p.requires_grad)
            model_info = dict(total_params=total, trainable_params=trainable, head_params=head,
                              head_trainable_params=head_trainable,
                              macs_per_image=macs_info(model, device), head_structure=config["head_structure"])
        logger.flush_all(config, model_info)
        save_curves(run_dir, pd.DataFrame(logger.tables["epochs"]), name)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    folds = pd.DataFrame(logger.tables["folds"])
    finished = sorted(int(x) for x in folds.fold.unique())
    status = "COMPLETE" if finished == list(split_folds) else f"PARTIAL folds {finished}"
    (run_dir / "STATUS.txt").write_text(f"{status}  ({dt.datetime.now():%Y-%m-%d %H:%M:%S})")
    log(f"\nRUN {name}: {status}")
    log(f"  test acc {pd.to_numeric(folds.test_acc).mean():.2f} +/- {pd.to_numeric(folds.test_acc).std(ddof=1) if len(folds) > 1 else 0:.2f} | "
        f"test macro-F1 {pd.to_numeric(folds.test_f1).mean():.4f}")
    return run_dir


if __name__ == "__main__":
    main()
