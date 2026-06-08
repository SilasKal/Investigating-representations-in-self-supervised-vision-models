"""
maps_train_test_by_instance_boxplot_runs.py

Create per-dataset plots showing per-instance accuracy (instances 1..5) for each model
where each box represents the distribution across multiple runs (e.g. 5 runs) for that
model-instance.

Usage:
    1) Optionally edit CONFIG values below or ensure MODEL_SPECS entries include a
       "runs" key with a list of checkpoint paths (one entry per run).
    2) Run: python maps_train_test_by_instance_boxplot_runs.py

Model spec examples supported:
    - {"name": "ssl_resnet", "runs": ["/abs/path/run1.pth", "/abs/path/run2.pth", ...]}
    - {"name": "supervised", "path": "/abs/path/single_checkpoint.pth"}  # treated as single run

If a model has fewer than 2 runs available a warning is printed and the model is still
plotted (box will reflect available runs). If no valid runs are found for a model it is
skipped.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from torchvision import transforms
from typing import cast

from maps_per_instance_accuracy import (
    DATASETS_PARENT_ROOT,
    DEVICE,
    EXPECTED_NUM_CLASSES,
    LIMIT_PER_INSTANCE,
    MAPS_INDICES_PATH,
    MODEL_SPECS,
    SUPERVISED_TEST_TF,
    TARGET_CLASS,
    TARGET_NAME,
    compute_per_instance_accuracy,
    discover_dataset_roots,
    load_maps_indices,
    resolve_target_class,
)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOT_ROOT = os.path.join(REPO_ROOT, "classification_performance_plots", "train_test_by_instance_runs")
SHOW_PLOT = False
NUM_EXPECTED_INSTANCES = 5
MODEL_FILES_ALL_RUNS_ROOT = os.path.join(REPO_ROOT, "model_files_all_runs")
RUNS_CSV_SUFFIX = "_by_instance_runs_long.csv"
SUMMARY_CSV_SUFFIX = "_by_instance_runs.csv"
USE_CSV_DEFAULT = False


def _discover_model_runs_from_directory(root_dir: str) -> List[Dict[str, object]]:
    """Dynamically discover all models in model_files_all_runs and generate MODEL_SPECS.

    Directory structure expected:
        model_files_all_runs/
          {resnet18, resnet50}/
            {pretrained, scratch}/
              {action, ssl, supervised}/
                {epoch_49_v1.pt, epoch_49_v2.pt, ...}

    Returns a list of model spec dicts with "name" and "runs" keys.
    """
    specs: List[Dict[str, object]] = []

    if not os.path.isdir(root_dir):
        print(f"Warning: model_files_all_runs directory not found at {root_dir}")
        return specs

    # Iterate: resnet18, resnet50
    for arch in sorted(os.listdir(root_dir)):
        arch_dir = os.path.join(root_dir, arch)
        if not os.path.isdir(arch_dir) or arch not in ("resnet18", "resnet50"):
            continue

        # Iterate: pretrained, scratch
        for pretrain in sorted(os.listdir(arch_dir)):
            pretrain_dir = os.path.join(arch_dir, pretrain)
            if not os.path.isdir(pretrain_dir) or pretrain not in ("pretrained", "scratch"):
                continue

        # Iterate: action, ssl, supervised
        for model_type in sorted(os.listdir(pretrain_dir)):
            model_type_dir = os.path.join(pretrain_dir, model_type)
            if not os.path.isdir(model_type_dir) or model_type not in ("action", "ssl", "supervised"):
                continue

            # Collect all .pt files (runs)
            run_files = sorted([f for f in os.listdir(model_type_dir) if f.endswith(".pt")])
            if not run_files:
                print(f"Warning: no .pt files found in {model_type_dir}")
                continue

            run_paths = [os.path.join(model_type_dir, f) for f in run_files]
            model_name = f"{arch}_{pretrain}_{model_type}"

            specs.append(
                {
                    "name": model_name,
                    "runs": run_paths,
                    "use_sup_lin_projector": model_type in ("supervised", "ssl", "action"),
                }
            )
            print(f"Discovered: {model_name} with {len(run_paths)} runs")

    return specs


def _model_order_key(model_name: str) -> tuple:
    """Sort models: resnet18 < resnet50, then pretrained < scratch, then supervised < ssl < action."""
    name = model_name.strip().lower()
    
    # Architecture order
    if "resnet18" in name:
        arch_order = 0
    elif "resnet50" in name:
        arch_order = 1
    else:
        arch_order = 2
    
    # Pretraining order
    if "pretrained" in name:
        pretrain_order = 0
    elif "scratch" in name:
        pretrain_order = 1
    else:
        pretrain_order = 2
    
    # Model type order
    if "supervised" in name:
        type_order = 0
    elif "ssl" in name:
        type_order = 1
    elif "action" in name:
        type_order = 2
    else:
        type_order = 3
    
    return (arch_order, pretrain_order, type_order, name)


def _gather_runs_for_spec(spec: Dict[str, object]) -> List[str]:
    """Return a list of absolute paths for runs for a model spec.

    - If spec contains 'runs' (iterable), use those paths.
    - Else if 'path' exists and points to a file, return single path.
    - Else return an empty list.
    """
    runs = []
    if "runs" in spec and spec["runs"]:
        for p in cast(List[object], spec["runs"]):
            p_abs = os.path.abspath(str(p))
            if os.path.isfile(p_abs):
                runs.append(p_abs)
            else:
                print(f"Warning: run path not found: {p_abs}")
    else:
        p = spec.get("path")
        if p:
            p_abs = os.path.abspath(str(p))
            if os.path.isfile(p_abs):
                runs.append(p_abs)
            else:
                print(f"Warning: single model path not found: {p_abs}")
    return runs


def _load_stats_from_csv(plot_root: str, dataset_name: str) -> pd.DataFrame:
    """Load per-dataset CSV produced by this script and reconstruct a stats DataFrame.

    Prefer the run-level CSV if present; fall back to summary CSV.
    """
    runs_csv_path = os.path.join(plot_root, f"{dataset_name}{RUNS_CSV_SUFFIX}")
    summary_csv_path = os.path.join(plot_root, f"{dataset_name}{SUMMARY_CSV_SUFFIX}")

    if os.path.isfile(runs_csv_path):
        df_long = pd.read_csv(runs_csv_path)
        rows: List[Dict[str, object]] = []
        for model, group in df_long.groupby("model"):
            # Build per-run matrix with fixed instance columns 1..NUM_EXPECTED_INSTANCES
            run_ids = sorted(group["run_idx"].unique().tolist())
            per_run_rows: List[np.ndarray] = []
            for run_idx in run_ids:
                run_group = group[group["run_idx"] == run_idx]
                inst_to_val = {int(r["instance"]): float(r["accuracy"]) for _, r in run_group.iterrows()}
                vals = [inst_to_val.get(i, float("nan")) for i in range(1, NUM_EXPECTED_INSTANCES + 1)]
                per_run_rows.append(np.asarray(vals, dtype=float))

            per_run_matrix = np.vstack(per_run_rows) if per_run_rows else np.empty((0, NUM_EXPECTED_INSTANCES))
            per_instance_lists = [per_run_matrix[:, i] for i in range(NUM_EXPECTED_INSTANCES)] if per_run_rows else []
            instance_means = [float(np.nanmean(v)) for v in per_instance_lists] if per_run_rows else []
            instance_stds = [float(np.nanstd(v, ddof=0)) for v in per_instance_lists] if per_run_rows else []

            rows.append(
                {
                    "model": model,
                    "instance_values": per_instance_lists,
                    "instance_means": instance_means,
                    "instance_stds": instance_stds,
                    "per_run_matrix": per_run_matrix,
                    "num_runs": int(per_run_matrix.shape[0]),
                }
            )

        if rows:
            return pd.DataFrame(rows)

    if not os.path.isfile(summary_csv_path):
        raise FileNotFoundError(f"Saved CSV not found: {summary_csv_path}")

    df = pd.read_csv(summary_csv_path)
    rows = []

    for model, group in df.groupby("model"):
        # Expect instances 1..NUM_EXPECTED_INSTANCES in group
        per_instance_lists: List[np.ndarray] = []
        per_instance_means: List[float] = []
        per_instance_stds: List[float] = []

        for inst in range(1, NUM_EXPECTED_INSTANCES + 1):
            row = group[group["instance"] == inst]
            if row.empty:
                # if missing, use nan
                lo = hi = mean = np.nan
                std = 0.0
            else:
                mean = float(row.iloc[0]["mean"])
                std = float(row.iloc[0]["std"])
                lo = float(np.clip(mean - std, 0.0, 1.0))
                hi = float(np.clip(mean + std, 0.0, 1.0))

            # Build pseudo-distribution of length 5
            vals = np.array([lo, mean, mean, mean, hi], dtype=float)
            per_instance_lists.append(vals)
            per_instance_means.append(float(mean))
            per_instance_stds.append(float(std))

        # per_run_matrix: shape (num_pseudo_runs, NUM_EXPECTED_INSTANCES)
        try:
            per_run_matrix = np.stack(per_instance_lists, axis=1)
        except Exception:
            per_run_matrix = np.empty((0, NUM_EXPECTED_INSTANCES))

        rows.append(
            {
                "model": model,
                "instance_values": per_instance_lists,
                "instance_means": per_instance_means,
                "instance_stds": per_instance_stds,
                "per_run_matrix": per_run_matrix,
                "num_runs": int(per_run_matrix.shape[0]),
            }
        )

    return pd.DataFrame(rows)


def _save_run_level_csv(stats_df: pd.DataFrame, plot_root: str, dataset_name: str) -> None:
    """Persist run-level per-instance accuracies for later plotting without recompute."""
    csv_rows: List[Dict[str, object]] = []
    for _, row in stats_df.iterrows():
        model_name = str(row["model"])
        per_run = row.get("per_run_matrix")
        if per_run is None or getattr(per_run, "size", 0) == 0:
            continue
        per_run = np.asarray(per_run, dtype=float)
        for run_idx in range(per_run.shape[0]):
            for inst_idx in range(per_run.shape[1]):
                csv_rows.append(
                    {
                        "model": model_name,
                        "run_idx": int(run_idx),
                        "instance": int(inst_idx + 1),
                        "accuracy": float(per_run[run_idx, inst_idx]),
                    }
                )

    if not csv_rows:
        print(f"Warning: no run-level rows to save for {dataset_name}")
        return

    os.makedirs(plot_root, exist_ok=True)
    runs_csv_path = os.path.join(plot_root, f"{dataset_name}{RUNS_CSV_SUFFIX}")
    pd.DataFrame(csv_rows).to_csv(runs_csv_path, index=False)
    print(f"Saved run-level CSV: {runs_csv_path}")


def _merge_stats_by_model(base_df: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge updated model rows into an existing stats DataFrame by model name."""
    if base_df is None or base_df.empty:
        return new_df.copy()
    if new_df is None or new_df.empty:
        return base_df.copy()

    updated_models = set(new_df["model"].tolist())
    base_keep = base_df[~base_df["model"].isin(updated_models)]
    return pd.concat([base_keep, new_df], ignore_index=True)


def _collect_model_runs_stats(
    dataset_root: str,
    target_class: int,
    model_specs: List[Dict[str, object]],
    device: Optional[str] = None,
    limit_per_instance: Optional[int] = None,
    expected_num_classes: Optional[int] = None,
) -> pd.DataFrame:
    """Collect per-instance accuracies across runs for each model.

    Returns a DataFrame with rows: {
        'model': name,
        'instance_values': [array_for_inst1, ..., array_for_inst5],
        'instance_means': [..],
        'instance_stds': [..]
    }
    """
    rows: List[Dict[str, object]] = []

    for spec in model_specs:
        model_name = str(spec.get("name", "model")).strip() or "model"
        runs = _gather_runs_for_spec(spec)
        if not runs:
            print(f"Skipping '{model_name}': no valid run checkpoints found in spec")
            continue

        print(f"Collecting runs for model '{model_name}' on dataset '{os.path.basename(dataset_root)}' ({len(runs)} runs)")

        # For each run, compute per-instance accuracy (expects results for instances in order)
        per_run_instance_acc: List[np.ndarray] = []

        for run_path in runs:
            try:
                results_df, _ = compute_per_instance_accuracy(
                    dataset_root=dataset_root,
                    model_name=run_path,
                    target_class=target_class,
                    device=device,
                    use_sup_lin_projector=bool(spec.get("use_sup_lin_projector", False)),
                    limit_per_instance=limit_per_instance,
                    model_label=model_name,
                    print_predictions_debug=False,
                    predictions_debug_preview=None,
                    expected_num_classes=expected_num_classes,
                    preprocess_override=(
                        SUPERVISED_TEST_TF
                        if "supervised" in model_name.lower()
                        else (SUPERVISED_TEST_TF if ("ssl" in model_name.lower() or "action" in model_name.lower()) else None)
                    ),
                )
            except Exception as e:
                print(f"Error evaluating run '{run_path}' for model '{model_name}': {e}")
                continue

            if results_df.empty:
                print(f"Warning: run '{run_path}' produced no instance results; skipping run")
                continue

            if len(results_df) < NUM_EXPECTED_INSTANCES:
                print(f"Warning: run '{run_path}' returned {len(results_df)} instances (expected {NUM_EXPECTED_INSTANCES}); skipping run")
                continue

            # Ensure ordering and take first NUM_EXPECTED_INSTANCES rows
            inst_acc = np.asarray(results_df.iloc[:NUM_EXPECTED_INSTANCES]["accuracy"].to_numpy(dtype=float))
            per_run_instance_acc.append(inst_acc)

        if not per_run_instance_acc:
            print(f"Skipping '{model_name}': no successful runs collected")
            continue

        # Build arrays per instance: list length NUM_EXPECTED_INSTANCES, each an array of len(num_runs)
        per_instance_lists: List[np.ndarray] = []
        arr = np.stack(per_run_instance_acc, axis=0)  # shape: (num_runs, NUM_EXPECTED_INSTANCES)
        for inst_idx in range(NUM_EXPECTED_INSTANCES):
            per_instance_lists.append(arr[:, inst_idx])

        instance_means = [float(np.nanmean(v)) for v in per_instance_lists]
        instance_stds = [float(np.nanstd(v, ddof=0)) for v in per_instance_lists]

        rows.append(
            {
                "model": model_name,
                "instance_values": per_instance_lists,
                "instance_means": instance_means,
                "instance_stds": instance_stds,
                "per_run_matrix": arr,
                "num_runs": int(arr.shape[0]),
            }
        )

    if not rows:
        raise RuntimeError("No valid model run results collected; check MODEL_SPECS and run checkpoint paths.")

    return pd.DataFrame(rows)


def plot_by_instance(
    all_data: Dict[str, pd.DataFrame],
    plot_root: str,
    show_plot: bool = False,
) -> None:
    """Create per-dataset plots: x-axis instances 1..5, box per model per instance from runs.
    """
    colors = {
        "supervised": "#1f77b4",
        "ssl": "#ff7f0e",
        "action": "#2ca02c",
    }
    default_color = "#9467bd"

    for dataset_name, stats_df in all_data.items():
        all_models = stats_df["model"].tolist()
        all_models = sorted(all_models, key=_model_order_key)

        model_colors = {}
        for model in all_models:
            model_lower = model.lower()
            for key, color in colors.items():
                if key in model_lower:
                    model_colors[model] = color
                    break
            if model not in model_colors:
                model_colors[model] = default_color

        instances = np.arange(1, NUM_EXPECTED_INSTANCES + 1)
        x_positions = np.arange(len(instances))
        box_width = 0.8 / max(1, len(all_models))

        fig, ax = plt.subplots(figsize=(12, 6))

        data_by_model = {model: stats_df[stats_df["model"] == model].iloc[0] for model in all_models}

        positions_by_model = {model: [] for model in all_models}
        data_for_boxplots_by_model = {model: [] for model in all_models}

        for inst_idx, inst in enumerate(instances):
            for model_idx, model in enumerate(all_models):
                row = data_by_model.get(model)
                if row is None:
                    continue
                inst_vals = row["instance_values"][inst_idx]
                # boxplot expects sequence of sequences
                data_for_boxplots_by_model[model].append(inst_vals)
                positions_by_model[model].append(
                    x_positions[inst_idx] + (model_idx - len(all_models) / 2) * box_width + box_width / 2
                )

        # Plot per-model box series
        for model in all_models:
            if not data_for_boxplots_by_model[model]:
                continue
            bp = ax.boxplot(
                data_for_boxplots_by_model[model],
                positions=positions_by_model[model],
                widths=box_width * 0.8,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                label=f"{model} (n={int(stats_df[stats_df['model']==model].iloc[0]['num_runs'])})",
                flierprops={
                    "marker": "o",
                    "markerfacecolor": model_colors[model],
                    "markeredgecolor": model_colors[model],
                    "markersize": 4,
                    "alpha": 0.9,
                },
            )

            for box in bp["boxes"]:
                box.set(facecolor=model_colors[model], alpha=0.6, edgecolor=model_colors[model], linewidth=0.8)
            for median in bp["medians"]:
                median.set(color=model_colors[model], linewidth=1.2)
            for whisker in bp["whiskers"]:
                whisker.set(color=model_colors[model], linewidth=0.8)
            for cap in bp["caps"]:
                cap.set(color=model_colors[model], linewidth=0.8)
            for mean in bp.get("means", []):
                mean.set(marker="o", markerfacecolor=model_colors[model], markeredgecolor=model_colors[model], markersize=4)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in instances])
        ax.set_xlabel("Instance (1..5)")
        ax.set_ylabel("Top-1 Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

        fig.tight_layout()
        os.makedirs(plot_root, exist_ok=True)
        out_path = os.path.join(plot_root, f"{dataset_name}_by_instance_runs.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved per-instance runs plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        # Also write CSV summary
        csv_rows: List[Dict[str, object]] = []
        for _, row in stats_df.iterrows():
            for inst_idx in range(NUM_EXPECTED_INSTANCES):
                vals = np.asarray(row["instance_values"][inst_idx], dtype=float)
                csv_rows.append(
                    {
                        "model": row["model"],
                        "instance": inst_idx + 1,
                        "num_runs": int(row["num_runs"]),
                        "mean": float(np.mean(vals)) if vals.size > 0 else float('nan'),
                        "std": float(np.std(vals, ddof=0)) if vals.size > 0 else float('nan'),
                    }
                )
        csv_path = os.path.join(plot_root, f"{dataset_name}{SUMMARY_CSV_SUFFIX}")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Saved CSV summary: {csv_path}")
        _save_run_level_csv(stats_df=stats_df, plot_root=plot_root, dataset_name=dataset_name)


def plot_by_instance_violin(
    all_data: Dict[str, pd.DataFrame],
    plot_root: str,
    show_plot: bool = False,
) -> None:
    """Create per-dataset violin plots: x-axis instances 1..5, per-model violin per instance."""
    colors = {
        "supervised": "#1f77b4",
        "ssl": "#ff7f0e",
        "action": "#2ca02c",
    }
    default_color = "#9467bd"

    for dataset_name, stats_df in all_data.items():
        all_models = stats_df["model"].tolist()
        all_models = sorted(all_models, key=_model_order_key)

        model_colors = {}
        for model in all_models:
            model_lower = model.lower()
            for key, color in colors.items():
                if key in model_lower:
                    model_colors[model] = color
                    break
            if model not in model_colors:
                model_colors[model] = default_color

        instances = np.arange(1, NUM_EXPECTED_INSTANCES + 1)
        x_positions = np.arange(len(instances))
        violin_width = 0.8 / max(1, len(all_models))

        fig, ax = plt.subplots(figsize=(12, 6))
        data_by_model = {model: stats_df[stats_df["model"] == model].iloc[0] for model in all_models}

        for inst_idx, _ in enumerate(instances):
            for model_idx, model in enumerate(all_models):
                row = data_by_model.get(model)
                if row is None:
                    continue
                inst_vals = np.asarray(row["instance_values"][inst_idx], dtype=float)
                pos = x_positions[inst_idx] + (model_idx - len(all_models) / 2) * violin_width + violin_width / 2
                if inst_vals.size == 0 or np.all(np.isnan(inst_vals)):
                    continue

                vp = ax.violinplot(
                    [inst_vals],
                    positions=[pos],
                    widths=violin_width * 0.9,
                    showmeans=False,
                    showextrema=True,
                    showmedians=True,
                )

                for body in vp["bodies"]:
                    body.set_facecolor(model_colors[model])
                    body.set_edgecolor(model_colors[model])
                    body.set_alpha(0.5)

                for part in ("cmins", "cmaxes", "cbars", "cmedians"):
                    if part in vp:
                        vp[part].set_color(model_colors[model])
                        vp[part].set_linewidth(0.8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in instances])
        ax.set_xlabel("Instance (1..5)")
        ax.set_ylabel("Top-1 Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Legend: one entry per model
        handles = []
        labels = []
        for model in all_models:
            handles.append(plt.Line2D([0], [0], color=model_colors[model], lw=4))
            labels.append(f"{model} (n={int(stats_df[stats_df['model']==model].iloc[0]['num_runs'])})")
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 1.0))

        fig.tight_layout()
        os.makedirs(plot_root, exist_ok=True)
        out_path = os.path.join(plot_root, f"{dataset_name}_by_instance_runs_violin.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved per-instance runs violin plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def plot_model_type_violin_overview(
    all_data: Dict[str, pd.DataFrame],
    plot_root: str,
    show_plot: bool = False,
) -> None:
    """Create violin plots by model type for all instances and train-only instances."""
    model_types = ["supervised", "ssl", "action", "other"]
    type_labels = {
        "supervised": "Supervised",
        "ssl": "SSL",
        "action": "Action",
        "other": "Other",
    }
    colors = {
        "supervised": "#3776ab",
        "ssl": "#ff7f0e",
        "action": "#2ca02c",
        "other": "#9467bd",
    }
 
    type_to_all: Dict[str, List[float]] = {t: [] for t in model_types}
    type_to_train: Dict[str, List[float]] = {t: [] for t in model_types}
    type_to_test: Dict[str, List[float]] = {t: [] for t in model_types}

    for stats_df in all_data.values():
        for _, row in stats_df.iterrows():
            model_name = str(row.get("model", ""))
            model_type = _model_type_from_name(model_name)
            per_run = row.get("per_run_matrix")
            if per_run is None or getattr(per_run, "size", 0) == 0:
                continue
            per_run = np.asarray(per_run, dtype=float)

            # All instances (1..5)
            type_to_all[model_type].extend(per_run.flatten().tolist())

            # Train-only instances (1..4)
            if per_run.shape[1] >= 4:
                type_to_train[model_type].extend(per_run[:, :4].flatten().tolist())
            else:
                type_to_train[model_type].extend(per_run.flatten().tolist())

            # Test-only instance (5)
            if per_run.shape[1] >= 5:
                type_to_test[model_type].extend(per_run[:, 4].flatten().tolist())
            else:
                type_to_test[model_type].extend(per_run[:, -1].flatten().tolist())

    def _plot_violin(type_to_vals: Dict[str, List[float]], out_name: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        present_types = [t for t in model_types if len(type_to_vals.get(t, [])) > 0]
        if not present_types:
            print(f"Warning: no data available for '{title}' — skipping")
            plt.close(fig)
            return

        data = [np.asarray(type_to_vals[t], dtype=float) for t in present_types]
        positions = np.arange(len(present_types))

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.8,
            showmeans=False,
            showextrema=True,
            showmedians=True,
        )

        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[present_types[i]])
            body.set_edgecolor(colors[present_types[i]])
            body.set_alpha(0.5)

        for part in ("cmins", "cmaxes", "cbars", "cmedians"):
            if part in vp:
                vp[part].set_color("#444444")
                vp[part].set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels([type_labels[t] for t in present_types], fontsize=20)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy", fontsize=25)
        ax.set_xlabel("Model type", fontsize=25)
        # ax.set_title(title)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        os.makedirs(plot_root, exist_ok=True)
        out_path = os.path.join(plot_root, out_name)
        fig.savefig(out_path, dpi=150)
        print(f"Saved model-type violin plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    _plot_violin(
        type_to_all,
        out_name="overview_model_type_all_instances_violin.png",
        title="Model Type Accuracy (All Instances 1..5)",
    )
    _plot_violin(
        type_to_train,
        out_name="overview_model_type_train_instances_violin.png",
        title="Model Type Accuracy (Train Instances 1..4)",
    )
    _plot_violin(
        type_to_test,
        out_name="overview_model_type_test_instance_violin.png",
        title="Model Type Accuracy (Test Instance 5)",
    )


def plot_overview_train_test(
    all_data: Dict[str, pd.DataFrame],
    plot_root: str,
    show_plot: bool = False,
) -> None:
    """Create two overview plots across object categories:
    - Training (instances 1-4): x-axis categories, y-axis accuracy distributions over runs (per-run mean across instances 1-4)
    - Testing (instance 5): x-axis categories, y-axis accuracy distributions over runs (instance 5 per run)
    """
    # Color mapping per request: supervised=green, ssl=orange, action=green
    colors = {
        "supervised": "#3776ab",
        "ssl": "#ff7f0e",
        "action": "#2ca02c",
    }
    default_color = "#9467bd"

    def _category_order_by_global_test_mean() -> List[str]:
        """Order categories by global mean test accuracy (instance 5) across all model types."""
        category_means: List[tuple] = []
        for category, stats_df in all_data.items():
            per_run_test_vals: List[float] = []
            for _, row in stats_df.iterrows():
                per_run = row.get("per_run_matrix")
                if per_run is None or getattr(per_run, "size", 0) == 0:
                    continue
                per_run = np.asarray(per_run, dtype=float)
                num_inst = per_run.shape[1]
                if num_inst == 0:
                    continue
                test_vals = per_run[:, 4] if num_inst >= 5 else per_run[:, -1]
                per_run_test_vals.extend(np.asarray(test_vals, dtype=float).tolist())

            mean_test = float(np.nanmean(per_run_test_vals)) if per_run_test_vals else float("nan")
            category_means.append((category, mean_test))

        # Sort lowest to highest; keep a stable tiebreaker by name.
        return [c for c, _ in sorted(category_means, key=lambda t: (np.nan_to_num(t[1], nan=float("inf")), t[0]))]

    category_order = _category_order_by_global_test_mean()

    # Helper to plot overview for a given pretraining category ('pretrained' or 'scratch')
    def _plot_for_pretrain(pretrain_key: str):
        # Collect all models for this pretrain_key across datasets
        all_models = set()
        for stats_df in all_data.values():
            all_models.update([m for m in stats_df["model"].tolist() if pretrain_key in m.lower()])

        if not all_models:
            print(f"No models found for pretraining='{pretrain_key}' — skipping overview plots")
            return

        # Order models by type first (supervised -> ssl -> action), then name
        def _type_order_key(name: str) -> tuple:
            n = name.lower()
            if "supervised" in n:
                t = 0
            elif "ssl" in n:
                t = 1
            elif "action" in n:
                t = 2
            else:
                t = 3
            return (t, n)

        all_models = sorted(list(all_models), key=_type_order_key)

        # assign colors per model (supervised=green, ssl=orange, action=green)
        model_colors = {}
        for model in all_models:
            model_lower = model.lower()
            for key, color in colors.items():
                if key in model_lower:
                    model_colors[model] = color
                    break
            if model not in model_colors:
                model_colors[model] = default_color

        category_names = category_order if category_order else sorted(all_data.keys())
        x_positions = np.arange(len(category_names))
        box_width = 0.8 / max(1, len(all_models))

        # TRAIN overview (instances 1-4)
        fig_train, ax_train = plt.subplots(figsize=(14, 6))

        train_data_by_model = {model: [] for model in all_models}
        train_positions_by_model = {model: [] for model in all_models}

        for cat_idx, category in enumerate(category_names):
            stats_df = all_data[category]
            for model_idx, model in enumerate(all_models):
                row = stats_df[stats_df["model"] == model]
                if row.empty:
                    continue
                per_run = row.iloc[0].get("per_run_matrix")
                # per_run shape: (num_runs, NUM_EXPECTED_INSTANCES)
                if per_run is None or getattr(per_run, 'size', 0) == 0:
                    continue
                per_run = np.asarray(per_run, dtype=float)
                # compute per-run mean across instances 0..3
                num_inst = per_run.shape[1]
                if num_inst >= 4:
                    train_per_run = per_run[:, :4].mean(axis=1)
                else:
                    train_per_run = per_run.mean(axis=1)

                train_data_by_model[model].append(train_per_run)
                train_positions_by_model[model].append(
                    x_positions[cat_idx] + (model_idx - len(all_models)/2) * box_width + box_width/2
                )

        for model in all_models:
            if train_data_by_model[model]:
                bp = ax_train.boxplot(
                    train_data_by_model[model],
                    positions=train_positions_by_model[model],
                    widths=box_width * 0.8,
                    patch_artist=True,
                    showmeans=False,
                    meanline=False,
                    label=model,
                )
                for box in bp["boxes"]:
                    box.set(facecolor=model_colors[model], alpha=0.6, edgecolor=model_colors[model], linewidth=0.8)
                for median in bp["medians"]:
                    median.set(color=model_colors[model], linewidth=1.2)

        ax_train.set_xticks(x_positions)
        ax_train.set_xticklabels(category_names, rotation=45, ha="right")
        ax_train.set_ylim(0.0, 1.05)
        ax_train.set_ylabel("Accuracy")
        ax_train.set_xlabel("Object Category")
        ax_train.grid(axis="y", alpha=0.3)
        fig_train.tight_layout()
        train_plot_path = os.path.join(plot_root, f"overview_{pretrain_key}_training_instances_1-4.png")
        fig_train.savefig(train_plot_path, dpi=150)
        print(f"Saved overview training plot: {train_plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig_train)

        # TEST overview (instance 5)
        fig_test, ax_test = plt.subplots(figsize=(14, 6))

        test_data_by_model = {model: [] for model in all_models}
        test_positions_by_model = {model: [] for model in all_models}

        for cat_idx, category in enumerate(category_names):
            stats_df = all_data[category]
            for model_idx, model in enumerate(all_models):
                row = stats_df[stats_df["model"] == model]
                if row.empty:
                    continue
                per_run = row.iloc[0].get("per_run_matrix")
                if per_run is None or getattr(per_run, 'size', 0) == 0:
                    continue
                per_run = np.asarray(per_run, dtype=float)
                num_inst = per_run.shape[1]
                if num_inst >= 5:
                    test_per_run = per_run[:, 4]
                else:
                    test_per_run = per_run[:, -1]

                test_data_by_model[model].append(test_per_run)
                test_positions_by_model[model].append(
                    x_positions[cat_idx] + (model_idx - len(all_models)/2) * box_width + box_width/2
                )

        for model in all_models:
            if test_data_by_model[model]:
                bp = ax_test.boxplot(
                    test_data_by_model[model],
                    positions=test_positions_by_model[model],
                    widths=box_width * 0.8,
                    patch_artist=True,
                    showmeans=False,
                    meanline=False,
                    label=model,
                )
                for box in bp["boxes"]:
                    box.set(facecolor=model_colors[model], alpha=0.6, edgecolor=model_colors[model], linewidth=0.8)
                for median in bp["medians"]:
                    median.set(color=model_colors[model], linewidth=1.2)

        ax_test.set_xticks(x_positions)
        ax_test.set_xticklabels(category_names, rotation=45, ha="right")
        ax_test.set_ylim(0.0, 1.05)
        ax_test.set_ylabel("Accuracy")
        ax_test.set_xlabel("Object Category")
        ax_test.grid(axis="y", alpha=0.3)
        fig_test.tight_layout()
        test_plot_path = os.path.join(plot_root, f"overview_{pretrain_key}_testing_instance_5.png")
        fig_test.savefig(test_plot_path, dpi=150)
        print(f"Saved overview testing plot: {test_plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig_test)

    # End of helper

    # Create separate overviews for pretrained and scratch
    _plot_for_pretrain("pretrained")
    _plot_for_pretrain("scratch")


def _model_type_from_name(model_name: str) -> str:
    """Map model name to broad model type used in printed summaries."""
    name = model_name.lower()
    if "supervised" in name:
        return "supervised"
    if "ssl" in name:
        return "ssl"
    if "action" in name:
        return "action"
    return "other"


def print_train_test_mean_summary_by_model_type(all_data: Dict[str, pd.DataFrame]) -> None:
    """Print mean accuracy over instances 1..5 (train+test) across all runs, grouped by model type."""
    type_to_values: Dict[str, List[float]] = {}
    type_labels = {
        "supervised": "Supervised",
        "ssl": "SSL",
        "action": "Action",
        "other": "Other",
    }
    for stats_df in all_data.values():
        for _, row in stats_df.iterrows():
            model_name = str(row.get("model", ""))
            model_type = _model_type_from_name(model_name)
            per_run = row.get("per_run_matrix")

            if per_run is None or getattr(per_run, "size", 0) == 0:
                continue

            per_run = np.asarray(per_run, dtype=float)
            # Average over all available instances (expected 1..5), then collect per-run values.
            per_run_mean = np.nanmean(per_run, axis=1)
            if model_type not in type_to_values:
                type_to_values[model_type] = []
            type_to_values[model_type].extend(per_run_mean.tolist())

    print("\n" + "=" * 80)
    print("Mean accuracy over train+test instances (1..5), aggregated over runs")
    print("=" * 80)

    if not type_to_values:
        print("No run-level data available for summary.")
        return

    for model_type in ("supervised", "ssl", "action", "other"):
        vals = np.asarray(type_to_values.get(model_type, []), dtype=float)
        if vals.size == 0:
            continue
        label = type_labels.get(model_type, model_type)
        print(f"{label:>10s}: mean={np.nanmean(vals):.4f} | std={np.nanstd(vals, ddof=0):.4f} | n={vals.size}")


def main() -> None:
    datasets_parent_root = os.path.abspath(DATASETS_PARENT_ROOT)
    maps_indices_path = os.path.abspath(MAPS_INDICES_PATH)

    parser = argparse.ArgumentParser(description="Generate per-instance and overview boxplots from runs or saved CSVs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-csv", dest="use_csv", action="store_true", help="Load data from saved CSVs instead of running evaluation")
    group.add_argument("--no-csv", dest="use_csv", action="store_false", help="Run evaluation instead of loading CSVs")
    parser.add_argument(
        "--eval-model",
        dest="eval_models",
        action="append",
        default=[],
        help="Only evaluate models whose names contain this substring; other models are loaded from CSV",
    )
    parser.set_defaults(use_csv=USE_CSV_DEFAULT)
    args = parser.parse_args()
    use_csv = bool(args.use_csv)
    eval_models = [m.strip().lower() for m in args.eval_models if str(m).strip()]
    base_load_csv = use_csv or bool(eval_models)
    print(f"Eval models: {eval_models}")

    # Discover models from model_files_all_runs directory (only needed for evaluation)
    model_specs = MODEL_SPECS
    if (not use_csv) or eval_models:
        print("=" * 80)
        print("Discovering models from model_files_all_runs...")
        print("=" * 80)
        model_specs = _discover_model_runs_from_directory(MODEL_FILES_ALL_RUNS_ROOT)
        if not model_specs:
            print("Warning: no models discovered from model_files_all_runs; falling back to imported MODEL_SPECS")
            model_specs = MODEL_SPECS

    dataset_roots = discover_dataset_roots(datasets_parent_root)
    class_to_idx = load_maps_indices(maps_indices_path)

    os.makedirs(PLOT_ROOT, exist_ok=True)

    print(f"\nFound {len(dataset_roots)} dataset roots in: {datasets_parent_root}")
    print(f"Using {len(model_specs)} discovered model specs\n")

    all_data: Dict[str, pd.DataFrame] = {}
    target_class_global = None

    for dataset_root in dataset_roots:
        dataset_name = os.path.basename(os.path.abspath(dataset_root))
        target_class = resolve_target_class(
            dataset_root=dataset_root,
            class_to_idx=class_to_idx,
            target_class=TARGET_CLASS,
            target_name=TARGET_NAME,
        )

        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Resolved target class: {target_class}")

        if target_class_global is None:
            target_class_global = target_class

        stats_df: Optional[pd.DataFrame] = None
        if base_load_csv:
            try:
                stats_df = _load_stats_from_csv(PLOT_ROOT, dataset_name)
            except FileNotFoundError:
                print(f"Saved CSV not found for dataset {dataset_name}; skipping")
                continue

        if eval_models:
            eval_specs = [
                spec
                for spec in model_specs
                if any(token in str(spec.get("name", "")).lower() for token in eval_models)
            ]
            if not eval_specs:
                print(f"Warning: no model specs matched --eval-model for dataset {dataset_name}; using CSV only")
            else:
                updated_df = _collect_model_runs_stats(
                    dataset_root=dataset_root,
                    target_class=target_class,
                    model_specs=eval_specs,
                    device=DEVICE,
                    limit_per_instance=LIMIT_PER_INSTANCE,
                    expected_num_classes=EXPECTED_NUM_CLASSES,
                )
                stats_df = _merge_stats_by_model(stats_df, updated_df)
        elif not use_csv:
            stats_df = _collect_model_runs_stats(
                dataset_root=dataset_root,
                target_class=target_class,
                model_specs=model_specs,
                device=DEVICE,
                limit_per_instance=LIMIT_PER_INSTANCE,
                expected_num_classes=EXPECTED_NUM_CLASSES,
            )
        if stats_df is None or stats_df.empty:
            print(f"Warning: no stats collected for dataset {dataset_name}; skipping")
            continue
        stats_df = stats_df.sort_values(by="model", key=lambda s: s.map(_model_order_key)).reset_index(drop=True)

        all_data[dataset_name] = stats_df

    print("\n" + "-" * 80)
    print("Creating per-dataset per-instance plots from runs...")
    plot_by_instance(all_data=all_data, plot_root=PLOT_ROOT, show_plot=SHOW_PLOT)
    print("Creating per-dataset per-instance violin plots from runs...")
    plot_by_instance_violin(all_data=all_data, plot_root=PLOT_ROOT, show_plot=SHOW_PLOT)
    # Also create overview plots across object categories (train instances 1-4, test instance 5)
    print("Creating overview plots across object categories...")
    plot_overview_train_test(all_data=all_data, plot_root= PLOT_ROOT, show_plot=SHOW_PLOT)
    print("Creating model-type violin overview plots...")
    plot_model_type_violin_overview(all_data=all_data, plot_root=PLOT_ROOT, show_plot=SHOW_PLOT)
    print_train_test_mean_summary_by_model_type(all_data=all_data)


if __name__ == "__main__":
    main()
