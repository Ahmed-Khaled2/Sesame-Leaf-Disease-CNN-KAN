"""
bench_speed.py - measured speed of every head, alone and inside the full model, on this PC.
(The paper's remarks ask for measured latency, not only MACs.)

For ResNet50 and EfficientNet-B0, and for each head (KAN, spline-only KAN, GP KAN without
LayerNorm, MLP 128, MLP 1536, linear) it measures:
  * full model, inference: GPU batch 1, GPU batch 64 (per photo), CPU batch 1
  * head only, inference:  GPU batch 1, GPU batch 64 (per photo), CPU batch 1
  * one training step (batch 64, last backbone part unfrozen, AdamW), GPU
  * parameters and MACs
Times are medians over many repeats, after a warm-up. Weights are the ImageNet ones (speed does
not depend on the trained values).

Run it when nothing else uses the GPU:     python bench_speed.py
In the phase 4 queue it is started with --wait: it first waits until the other workers are done.

Output: outputs/speed/speed.csv, speed.xlsx (+ the machine details)
"""
import argparse
import platform
import statistics
import time

import pandas as pd
import torch

import train
from analysis_common import wait_for_other_workers
from common import OUT_ROOT, environment_info

HEADS = [("KAN", ["--head", "kan"]),
         ("spline-only KAN", ["--head", "kan", "--kan-base", "off"]),
         ("KAN, GP head (no LayerNorm)", ["--head", "kan", "--norm", "none"]),
         ("MLP 128", ["--head", "mlp"]),
         ("MLP 1536 (same size as KAN)", ["--head", "mlp", "--hidden", "1536"]),
         ("linear", ["--head", "linear"])]
BACKBONES = ["resnet50", "efficientnet_b0"]


def timed(fn, reps, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out)


def measure(backbone, head_label, head_args, reps, has_cuda):
    a = train.get_args(["--data", "multi", "--backbone", backbone] + head_args)
    model = train.HybridNet(a, 4)
    total, trainable, head = train.count_params(model)
    row = dict(backbone=backbone, head=head_label, total_params=total, trainable_params=trainable, head_params=head,
               head_trainable_params=sum(p.numel() for p in model.head_params() if p.requires_grad))
    feat_dim = train.FEATURE_DIM[backbone]
    model.eval()
    x1, f1 = torch.randn(1, 3, 224, 224), torch.randn(1, feat_dim)
    with torch.inference_mode():                     # ---- CPU
        row["full_cpu_b1_ms"] = timed(lambda: model(x1), max(10, reps // 5), 3, False)
        row["head_cpu_b1_ms"] = timed(lambda: model.head(f1), reps * 4, 20, False)
    if has_cuda:                                     # move BEFORE inference_mode, so training still works
        model.cuda()
        x1g, x64g = x1.cuda(), torch.randn(64, 3, 224, 224, device="cuda")
        f1g, f64g = f1.cuda(), torch.randn(64, feat_dim, device="cuda")
        with torch.inference_mode():                 # ---- GPU
            row["full_gpu_b1_ms"] = timed(lambda: model(x1g), reps, 20, True)
            row["full_gpu_b64_ms_per_photo"] = timed(lambda: model(x64g), max(10, reps // 5), 3, True) / 64
            row["head_gpu_b1_ms"] = timed(lambda: model.head(f1g), reps * 4, 50, True)
            row["head_gpu_b64_ms_per_photo"] = timed(lambda: model.head(f64g), reps * 4, 50, True) / 64
    if has_cuda:
        model.set_train_mode()
        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        groups = [{"params": model.head_params(), "lr": 1e-3}] + ([{"params": bb, "lr": 1e-5}] if bb else [])
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
        x64 = torch.randn(64, 3, 224, 224, device="cuda")
        y64 = torch.randint(0, 4, (64,), device="cuda")

        def step():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x64), y64)
            loss.backward()
            opt.step()
        torch.cuda.reset_peak_memory_stats()
        row["train_step_gpu_b64_ms"] = timed(step, max(10, reps // 5), 3, True)
        row["train_step_gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    macs = train.macs_info(model, "cuda" if has_cuda else "cpu")
    row["macs_per_photo"] = macs
    del model
    if has_cuda:
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait", action="store_true", help="wait until the other queue workers have finished")
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    if args.wait:
        wait_for_other_workers()
    has_cuda = torch.cuda.is_available()
    out_dir = OUT_ROOT / "speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bb in BACKBONES:
        for label, head_args in HEADS:
            print(f"measuring {bb} + {label} ...", flush=True)
            try:
                rows.append(measure(bb, label, head_args, args.reps, has_cuda))
            except Exception as e:                    # noqa: BLE001 - keep going with the others
                print(f"   failed: {type(e).__name__}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "speed.csv", index=False)
    env = dict(environment_info(), cpu=platform.processor(), torch_threads=torch.get_num_threads(),
               gpu=torch.cuda.get_device_name(0) if has_cuda else "none", reps=args.reps,
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with pd.ExcelWriter(out_dir / "speed.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="speed", index=False)
        pd.DataFrame([dict(item=k, value=str(v)) for k, v in env.items()]).to_excel(xw, sheet_name="machine", index=False)
    cols = [c for c in ["backbone", "head", "head_params", "full_gpu_b1_ms", "full_cpu_b1_ms", "head_gpu_b1_ms",
                        "head_cpu_b1_ms", "train_step_gpu_b64_ms"] if c in df]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"saved {out_dir / 'speed.xlsx'}")


if __name__ == "__main__":
    main()
