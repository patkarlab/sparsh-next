#!/usr/bin/env python3
"""
SPARSH-next training: nested cross-validation, fold models, optional final model.

Example (defaults are the recommended recipe):
    python scripts/train.py --data_path /path/to/methylation.pkl --output_dir runs/next_default \
        --exclude_prefixes MPAL AML_NOS B-ALL_NOS

Reproduce the v0.1.0 recipe under the same fair evaluation, for comparison:
    python scripts/train.py ... --imbalance legacy_upsample \
        --train_sim mask --val_sim mask --coverage_mode schedule

Everything the run needs later (config, class and label maps, CpG list, fold
models, temperatures) is written to --output_dir. An existing non-empty
output directory is never overwritten unless --overwrite is given.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.dataset import encode_labels, filter_classes, load_label_map, load_training_data, subset  # noqa: E402
from evaluation.metrics import confusion_frame, predictions_frame, recall_by_class, summarize_probs  # noqa: E402
from models.corruption import COVERAGE_DISTS, COVERAGE_MODES, READ_SIMS, SIMULATIONS  # noqa: E402
from models.dilution import normal_index  # noqa: E402
from models.sparse_nn import ENCODINGS, softmax_np  # noqa: E402
from training.reproducibility import generate_run_id, get_environment_metadata, set_deterministic_mode  # noqa: E402
from training.trainer import (IMBALANCE_MODES, TrainConfig, cross_validate,  # noqa: E402
                              evaluation_conditions, train_final_model)
from training.training_curves import save_history  # noqa: E402

VERSION = (REPO / "VERSION").read_text().strip()
logger = logging.getLogger("sparsh")


def parse_args():
    p = argparse.ArgumentParser(description=f"SPARSH-next {VERSION}: nested CV training",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--data_path", required=True, help="Training pickle (samples x CpGs, ANNOTATION column)")
    d.add_argument("--output_dir", required=True, help="New directory for this run")
    d.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output_dir")
    d.add_argument("--label_map", default=str(REPO / "configs" / "label_map.json"),
                   help="Label merges; 'none' disables")
    d.add_argument("--exclude_ids", "--junk_path", dest="exclude_ids", default=None,
                   help="Text file of Sample_IDs to leave out")
    d.add_argument("--cpg_list", default=None, help="JSON list fixing which CpGs are used, in order")
    d.add_argument("--group_col", default=None, help="Column holding a patient or cluster ID for grouped CV")
    d.add_argument("--groups_file", default=None, help="CSV Sample_ID,group (e.g. from scripts/check_data.py)")
    d.add_argument("--min_samples", type=int, default=5)
    d.add_argument("--exclude_classes", nargs="*", default=[], help="Exact class names (after the label map)")
    d.add_argument("--exclude_prefixes", nargs="*", default=[], help="Drop every class starting with these")

    m = p.add_argument_group("model")
    m.add_argument("--hidden_dims", type=int, nargs="+", default=[1024, 512, 256])
    m.add_argument("--dropout", type=float, default=0.45)
    m.add_argument("--input_encoding", choices=ENCODINGS, default="midpoint")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--learning_rate", type=float, default=1e-4)
    t.add_argument("--weight_decay", type=float, default=1e-4)
    t.add_argument("--early_stopping_patience", type=int, default=30)
    t.add_argument("--lr_scheduler_patience", type=int, default=10)
    t.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    t.add_argument("--samples_per_class_per_batch", type=int, default=5)
    t.add_argument("--focal_gamma", type=float, default=2.0)
    t.add_argument("--label_smoothing", type=float, default=0.05)
    t.add_argument("--imbalance", choices=IMBALANCE_MODES, default="sampler")

    s = p.add_argument_group("sparsity simulation")
    s.add_argument("--train_sim", choices=SIMULATIONS, default="reads")
    s.add_argument("--coverage_mode", choices=COVERAGE_MODES, default="random")
    s.add_argument("--coverage_dist", choices=COVERAGE_DISTS, default="loguniform",
                   help="random mode: how the observed fraction is drawn between cov_min and cov_max")
    s.add_argument("--cov_min", type=float, default=0.02, help="random mode: lowest observed fraction")
    s.add_argument("--cov_max", type=float, default=0.5, help="random mode: highest observed fraction")
    s.add_argument("--mask_start", type=float, default=0.97, help="schedule mode: first-epoch masked fraction")
    s.add_argument("--mask_end", type=float, default=0.80, help="schedule mode: last-epoch masked fraction")
    s.add_argument("--call_error_max", type=float, default=0.0,
                   help="Per-read call error rate drawn per training sample, uniform on [0, this]; 0 = none. "
                        "Read simulations only (see models/corruption.py)")
    s.add_argument("--dilution_prob", type=float, default=0.0,
                   help="Share of leukaemia training samples diluted with a normal-marrow array before the reads "
                        "are simulated (see models/dilution.py); 0 = none")
    s.add_argument("--blast_min", type=float, default=0.2,
                   help="Diluted samples get a blast fraction drawn uniformly from [this, 1]")
    s.add_argument("--normal_class", default="Normal_Control_BM",
                   help="Class whose arrays are the dilution partners (never diluted themselves)")

    v = p.add_argument_group("validation and evaluation")
    v.add_argument("--n_folds", type=int, default=5)
    v.add_argument("--inner_val_frac", type=float, default=0.15)
    v.add_argument("--val_sim", choices=SIMULATIONS, default=None, help="Default: same as --train_sim")
    v.add_argument("--val_coverages", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    v.add_argument("--eval_coverages", type=float, nargs="+", default=[0.03, 0.05, 0.1, 0.2, 0.3])
    v.add_argument("--eval_sims", choices=SIMULATIONS, nargs="+", default=["reads", "mask"],
                   help="Simulations scored on the outer folds, each at every --eval_coverages")
    v.add_argument("--val_call_error", type=float, default=0.0,
                   help="Per-read call error rate of the inner validation sets (early stopping, temperature)")
    v.add_argument("--val_dilution", action="store_true",
                   help="Dilute the inner validation sets as training does (once per sample, seeded)")
    v.add_argument("--eval_blasts", type=float, nargs="+", default=[1.0],
                   help="Blast fractions at which the read simulations are scored; 1 gives the condition names "
                        "and inputs of earlier runs, 0.3 adds rows such as binary-blast30_0.30 (every leukaemia "
                        "sample of the outer fold diluted to 30%% blasts with a normal marrow from the training part)")
    v.add_argument("--eval_call_errors", type=float, nargs="+", default=[0.0],
                   help="Per-read call error rates at which the read simulations are scored; 0 gives the "
                        "condition names and inputs of earlier runs, e.g. binary_0.30; 0.1 adds binary-err10_0.30")
    v.add_argument("--eval_seed", type=int, default=12345, help="Keep fixed so runs are scored on identical inputs")
    v.add_argument("--no_calibration", action="store_true", help="Skip temperature scaling")
    v.add_argument("--threshold", type=float, default=0.90, help="Confidence threshold for callable metrics")
    v.add_argument("--primary_condition", default="auto",
                   help="Condition for confusion matrices; auto = first --eval_sims at the coverage nearest 0.2")

    o = p.add_argument_group("outputs and hardware")
    o.add_argument("--no_fold_models", action="store_true", help="Do not save fold weights (saves disk)")
    o.add_argument("--final_model", action="store_true",
                   help="Also train one model on all samples except an inner split used for early stopping")
    o.add_argument("--device", default="cuda")
    o.add_argument("--data_on_gpu", choices=["auto", "always", "never"], default="auto")
    o.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_output(args) -> Path:
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"Output directory {out} is not empty. Choose a new --output_dir or pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(out / "training_log.txt")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return out


def main():
    args = parse_args()
    out = setup_output(args)
    run_id = generate_run_id(args.seed)
    logger.info(f"SPARSH-next {VERSION} | run {run_id}")
    logger.info(f"Arguments: {vars(args)}")
    set_deterministic_mode(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA was requested but no GPU is visible. Submit to a GPU node, or pass --device cpu "
                 "deliberately (full-size training on CPU would exceed the walltime).")
    for cov in list(args.val_coverages) + list(args.eval_coverages) + [args.cov_min, args.cov_max]:
        if not 0.0 < cov < 1.0:
            sys.exit(f"Coverages must be observed fractions between 0 and 1 (got {cov})")
    for err in [args.call_error_max, args.val_call_error] + list(args.eval_call_errors):
        if not 0.0 <= err < 0.5:
            sys.exit(f"Call error rates must be between 0 and 0.5 (got {err}); 0.5 would make every read random")
    val_sim = args.val_sim or args.train_sim
    if not 0.0 <= args.dilution_prob <= 1.0:
        sys.exit(f"--dilution_prob must be between 0 and 1 (got {args.dilution_prob})")
    if not 0.0 < args.blast_min <= 1.0:
        sys.exit(f"--blast_min must be above 0 and at most 1 (got {args.blast_min})")
    for blast in args.eval_blasts:
        if not 0.0 < blast <= 1.0:
            sys.exit(f"--eval_blasts must be above 0 and at most 1 (got {blast})")
    if args.val_dilution and args.dilution_prob <= 0:
        sys.exit("--val_dilution needs --dilution_prob above 0")
    if args.call_error_max > 0 and args.train_sim not in READ_SIMS:
        sys.exit(f"--call_error_max needs a read simulation for --train_sim {READ_SIMS}, not {args.train_sim}")
    if args.val_call_error > 0 and val_sim not in READ_SIMS:
        sys.exit(f"--val_call_error needs a read simulation for --val_sim {READ_SIMS}, not {val_sim}")
    conditions = evaluation_conditions(args.eval_coverages, args.eval_sims, args.eval_call_errors, args.eval_blasts)
    condition_names = [c.name for c in conditions]
    if args.primary_condition == "auto":
        sim_rows = [c for c in conditions if c.sim == args.eval_sims[0] and c.call_error == 0.0 and c.blast == 1.0]
        sim_rows = sim_rows or [c for c in conditions if c.sim == args.eval_sims[0]]
        args.primary_condition = min(sim_rows, key=lambda c: abs(c.fraction - 0.2)).name if sim_rows else "dense"
    if args.primary_condition not in condition_names:
        sys.exit(f"--primary_condition {args.primary_condition} is not one of {condition_names}")

    # ---------------------------------------------------------------- data
    label_map_path = None if str(args.label_map).lower() == "none" else args.label_map
    label_map = load_label_map(label_map_path)
    bundle = load_training_data(args.data_path, label_map, args.exclude_ids, args.cpg_list,
                                args.group_col, args.groups_file)
    keep = filter_classes(bundle["labels"], args.min_samples, args.exclude_classes, args.exclude_prefixes)
    bundle = subset(bundle, keep)
    y, idx_to_class = encode_labels(bundle["labels"])
    X, sample_ids, groups = bundle["X"], bundle["sample_ids"], bundle["groups"]
    n_classes = len(idx_to_class)
    normal_idx = normal_index(idx_to_class, args.normal_class)
    needs_normals = args.dilution_prob > 0 or any(b < 1.0 for b in args.eval_blasts)
    if needs_normals and normal_idx is None:
        sys.exit(f"Dilution needs the class {args.normal_class!r} (--normal_class), which this training set lacks")
    counts = np.bincount(y, minlength=n_classes)
    logger.info(f"Training set: {len(y)} samples, {X.shape[1]} CpGs, {n_classes} classes")
    for i in range(n_classes):
        logger.info(f"  {i:>2} {idx_to_class[i]}: {counts[i]}")
    if counts.min() < args.n_folds:
        logger.warning(f"Some classes have fewer samples than folds ({counts.min()} < {args.n_folds})")

    with open(out / "class_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    with open(out / "label_map.json", "w") as f:
        json.dump({"merge": label_map}, f, indent=2)
    with open(out / "selected_cpgs.json", "w") as f:
        json.dump(bundle["cpg_ids"], f)
    with open(out / "environment.json", "w") as f:
        json.dump(get_environment_metadata(), f, indent=2, default=str)

    # ---------------------------------------------------------------- config
    model_config = {
        "input_dim": int(X.shape[1]), "hidden_dims": list(args.hidden_dims), "n_classes": n_classes,
        "dropout": args.dropout, "activation": "gelu", "input_encoding": args.input_encoding,
    }
    cfg = TrainConfig(
        epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience, lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor, samples_per_class_per_batch=args.samples_per_class_per_batch,
        focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing, imbalance=args.imbalance,
        train_sim=args.train_sim, coverage_mode=args.coverage_mode, coverage_dist=args.coverage_dist,
        cov_min=args.cov_min, cov_max=args.cov_max,
        mask_start=args.mask_start, mask_end=args.mask_end,
        call_error_max=args.call_error_max, val_call_error=args.val_call_error,
        eval_call_errors=tuple(dict.fromkeys(float(e) for e in args.eval_call_errors)), n_folds=args.n_folds,
        dilution_prob=args.dilution_prob, blast_min=args.blast_min, normal_class=args.normal_class,
        val_dilution=args.val_dilution, eval_blasts=tuple(dict.fromkeys(float(b) for b in args.eval_blasts)),
        inner_val_frac=args.inner_val_frac, val_sim=val_sim,
        val_coverages=tuple(args.val_coverages), eval_coverages=tuple(args.eval_coverages),
        eval_sims=tuple(args.eval_sims),
        eval_seed=args.eval_seed, calibrate=not args.no_calibration,
        clip_observed=(0.05, 0.95) if args.train_sim == "mask" else None,
        seed=args.seed, device=args.device, data_on_gpu=args.data_on_gpu,
    )
    logger.info(f"Model: {model_config}")
    logger.info(f"Training: {cfg.to_dict()}")

    n_params = sum(a * b + b for a, b in zip([X.shape[1]] + list(args.hidden_dims),
                                             list(args.hidden_dims) + [n_classes]))
    n_saved = (0 if args.no_fold_models else args.n_folds) + (1 if args.final_model else 0)
    need_gb = n_saved * n_params * 4 * 1.05 / 1e9
    free_gb = shutil.disk_usage(out).free / 1e9
    logger.info(f"Model weights to save: {n_saved} x {n_params * 4 / 1e9:.2f} GB; free space {free_gb:.0f} GB")
    if need_gb > free_gb:
        sys.exit(f"Not enough disk space in {out} for the model weights ({need_gb:.1f} GB needed, "
                 f"{free_gb:.1f} GB free). Point RUNS_DIR at a larger disk or use --no_fold_models.")

    # ---------------------------------------------------------------- nested CV
    history = []
    fold_dir = None if args.no_fold_models else out / "fold_models"
    cv = cross_validate(X, y, sample_ids, groups, n_classes, model_config, cfg, fold_dir, history, normal_idx)
    save_history(history, out)
    pd.DataFrame(cv["records"]).to_csv(out / "folds_summary.csv", index=False)
    pd.DataFrame({
        "sample_id": sample_ids, "label": [idx_to_class[i] for i in y],
        "group": groups if groups is not None else sample_ids, "outer_fold": cv["fold_of"] + 1,
    }).to_csv(out / "folds.csv", index=False)

    rows_cal, rows_raw, recalls = [], [], {}
    true_names = [idx_to_class[i] for i in y]
    for name, sim, frac, err, blast in cv["conditions"]:
        logits = cv["logits"][name]
        probs_raw = softmax_np(logits)
        probs = softmax_np(logits / cv["temperature_of"][:, None])  # each sample uses its own fold's temperature
        base = {"condition": name, "simulation": sim or "none", "observed_fraction": frac if frac else 1.0,
                "call_error": err, "blast": blast}
        rows_cal.append({**base, **summarize_probs(y, probs, args.threshold)})
        rows_raw.append({**base, **summarize_probs(y, probs_raw, args.threshold)})
        recalls[name] = recall_by_class(y, probs, n_classes)
        pred_df = predictions_frame(sample_ids, probs, idx_to_class, true_names,
                                    extra={"outer_fold": cv["fold_of"] + 1, "temperature": cv["temperature_of"]})
        pred_df.to_csv(out / f"cv_predictions_{name}.csv", index=False, float_format="%.5f")
        if name in ("dense", args.primary_condition):
            labels = [idx_to_class[i] for i in range(n_classes)]
            confusion_frame(true_names, pred_df["prediction"], labels).to_csv(out / f"cv_confusion_{name}.csv")

    metrics = pd.DataFrame(rows_cal)
    metrics.to_csv(out / "cv_metrics_by_condition.csv", index=False, float_format="%.4f")
    pd.DataFrame(rows_raw).to_csv(out / "cv_metrics_by_condition_uncalibrated.csv", index=False, float_format="%.4f")
    recall_df = pd.DataFrame(recalls, index=[idx_to_class[i] for i in range(n_classes)])
    recall_df.insert(0, "n_samples", counts)
    recall_df.index.name = "class"
    recall_df.to_csv(out / "cv_recall_by_class.csv", float_format="%.3f")

    # ---------------------------------------------------------------- final model (optional)
    final_info = None
    if args.final_model:
        result = train_final_model(X, y, sample_ids, groups, n_classes, model_config, cfg, history, normal_idx)
        torch.save(result["model"].state_dict(), out / "model.pt")
        final_info = {k: result[k] for k in ("best_epoch", "epochs_run", "best_inner_val_nll", "temperature")}
        save_history(history, out)

    config = {
        "sparsh_version": VERSION,
        "run_id": run_id,
        "model": model_config,
        "training": cfg.to_dict(),
        "data": {
            "data_path": str(args.data_path), "label_map": label_map_path, "exclude_ids": args.exclude_ids,
            "cpg_list": args.cpg_list, "group_col": args.group_col, "groups_file": args.groups_file,
            "min_samples": args.min_samples, "exclude_classes": args.exclude_classes,
            "exclude_prefixes": args.exclude_prefixes, "n_samples": int(len(y)), "n_cpgs": int(X.shape[1]),
            "class_counts": {idx_to_class[i]: int(counts[i]) for i in range(n_classes)},
        },
        "inference": {
            "threshold": args.threshold,
            "clip_observed": list(cfg.clip_observed) if cfg.clip_observed else None,
            "fold_models": None if args.no_fold_models else [
                {"file": f"fold_models/fold{r['fold']}.pt", "temperature": r["temperature"]} for r in cv["records"]],
            "final_model": None if final_info is None else {"file": "model.pt", **final_info},
        },
        "cv_folds": cv["records"],
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---------------------------------------------------------------- summary
    shown = metrics[["condition", "accuracy", "balanced_accuracy", "top2_accuracy",
                     f"callable_share_{args.threshold:.2f}", f"accuracy_callable_{args.threshold:.2f}", "ece"]]
    logger.info("Outer-fold results (each fold scored once; calibrated probabilities):\n"
                + shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    logger.info(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
