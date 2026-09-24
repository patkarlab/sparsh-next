#!/usr/bin/env python3
"""
Time the SPARSH-next training loop at full size on this node's GPU.

Random numbers stand in for the data: the training pickle is not read, no model
is trained on real samples, and nothing is saved. The script prints the GPU and
the PyTorch build, the seconds per training epoch (default recipe: balanced
batches, read-level simulation, 29 classes, 354,551 CpGs), the inner-validation
pass, the CPU time of the evaluation simulation, peak GPU memory, and the
estimated time of one five-fold run.

Submitted to each GPU queue by jobs/probe_gpus.sh. By hand, on a GPU node:
    python scripts/gpu_benchmark.py
"""

import argparse
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.corruption import corrupt_rows, corrupt_torch, sample_observed_fraction  # noqa: E402
from models.sparse_nn import SparseNN, predict_logits  # noqa: E402
from training.reproducibility import set_deterministic_mode  # noqa: E402
from training.trainer import BalancedBatchSampler, FocalLoss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_cpgs", type=int, default=354551)
    p.add_argument("--n_classes", type=int, default=29)
    p.add_argument("--largest_class", type=int, default=254,
                   help="training rows of the largest class in one fold (about 370 x 0.8 x 6/7)")
    p.add_argument("--smallest_class", type=int, default=4)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[1024, 512, 256])
    p.add_argument("--samples_per_class_per_batch", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3, help="timed epochs, after one warm-up epoch")
    p.add_argument("--n_inner", type=int, default=302, help="inner-validation rows per validation coverage")
    p.add_argument("--n_val_coverages", type=int, default=4)
    p.add_argument("--n_test", type=int, default=530, help="outer-fold rows scored per evaluation condition")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--device", default="cuda", help="'cpu' only to test this script")
    return p.parse_args()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    args = parse_args()
    device = args.device
    print(f"Node {socket.gethostname()} | Python {platform.python_version()} | PyTorch {torch.__version__} "
          f"(CUDA {torch.version.cuda})")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            sys.exit("RESULT: no GPU visible to PyTorch on this node")
        props = torch.cuda.get_device_properties(0)
        capability = f"sm_{props.major}{props.minor}"
        kernels = torch.cuda.get_arch_list()
        print(f"GPU {props.name} | {props.total_memory / 1e9:.0f} GB | {capability} | "
              f"PyTorch kernels: {' '.join(kernels)}")
        if capability not in kernels:
            print(f"Note: this PyTorch build has no kernels compiled for {capability}")

    set_deterministic_mode(42)  # the same settings as the training jobs
    sizes = np.maximum(np.round(np.geomspace(args.largest_class, args.smallest_class, args.n_classes)), 1)
    y = np.repeat(np.arange(args.n_classes), sizes.astype(int))
    n_fit = len(y)
    X = torch.rand((n_fit, args.n_cpgs), device=device)  # training matrix kept on the GPU, as in real runs
    y_t = torch.from_numpy(y).to(device)
    sampler = BalancedBatchSampler(y, args.samples_per_class_per_batch, seed=42)
    model = SparseNN(args.n_cpgs, args.hidden_dims, args.n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalLoss(2.0, None, 0.05).to(device)
    inner = [np.random.default_rng(i).random((args.n_inner, args.n_cpgs), dtype=np.float32)
             for i in range(args.n_val_coverages)]
    print(f"Fold size: {n_fit} training rows, {args.n_classes} classes, {args.n_cpgs} CpGs; "
          f"{len(sampler)} batches of {args.n_classes * args.samples_per_class_per_batch} per epoch")

    def train_epoch(epoch: int) -> None:
        model.train()
        sampler.set_epoch(epoch)
        for batch in sampler:
            idx = torch.as_tensor(batch, device=device)
            xb = X.index_select(0, idx)
            yb = y_t.index_select(0, idx)
            frac = sample_observed_fraction(len(batch), "random", epoch, 300, 0.02, 0.5, 0.97, 0.80, device)
            xb = corrupt_torch(xb, frac, "reads")
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            float(loss.item()), int((logits.argmax(1) == yb).sum().item()), float(frac.sum().item())

    try:
        t0 = time.time()
        train_epoch(0)
        sync(device)
        warmup = time.time() - t0
        times = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_epoch(epoch)
            sync(device)
            times.append(time.time() - t0)
        t0 = time.time()
        for Xs in inner:
            predict_logits(model, Xs, device, 256)
        sync(device)
        val_s = time.time() - t0
    except RuntimeError as err:
        sys.exit(f"RESULT: FAILED on this GPU: {err}")

    train_s = float(np.median(times))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else float("nan")

    # CPU side: seeded per-sample simulation used for the fixed evaluation inputs
    Xc = np.random.default_rng(1).random((50, args.n_cpgs), dtype=np.float32)
    ids = np.array([f"S{i}" for i in range(50)])
    t0 = time.time()
    corrupt_rows(Xc, ids, 0.2, "reads", 12345, salt=2)
    reads_row = (time.time() - t0) / 50
    t0 = time.time()
    corrupt_rows(Xc, ids, 0.2, "mask", 12345, salt=2)
    mask_row = (time.time() - t0) / 50

    n_cond = 5  # coverages per simulation in the default evaluation
    eval_cpu = args.n_test * n_cond * (reads_row + mask_row) + args.n_val_coverages * args.n_inner * reads_row
    eval_fwd = val_s * (2 * n_cond + 1) * args.n_test / (args.n_val_coverages * args.n_inner)
    per_epoch = train_s + val_s

    def run_hours(epochs_per_fold: int) -> float:
        return args.n_folds * (epochs_per_fold * per_epoch + eval_cpu + eval_fwd) / 3600

    legacy_extra_gb = (args.n_classes * args.largest_class - n_fit) * args.n_cpgs * 4 / 1e9
    print(f"Warm-up epoch {warmup:.1f} s; training epoch {train_s:.2f} s (median of {len(times)}); "
          f"inner-validation pass {val_s:.2f} s")
    print(f"Evaluation simulation on the CPU: {1000 * reads_row:.0f} ms per sample (reads), "
          f"{1000 * mask_row:.0f} ms (mask); about {(eval_cpu + eval_fwd) / 60:.1f} min per fold")
    print(f"Peak GPU memory {peak_gb:.1f} GB (default recipe); legacy keeps about {legacy_extra_gb:.0f} GB more "
          f"on the GPU for its upsampled rows")
    print(f"RESULT: one {args.n_folds}-fold run takes about {run_hours(100):.1f} h at 100 epochs per fold "
          f"and at most {run_hours(300):.1f} h at the 300-epoch cap (plus a few minutes to load the data)")


if __name__ == "__main__":
    main()
