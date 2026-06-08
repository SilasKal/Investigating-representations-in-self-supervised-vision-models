"""
maps_train_test_transformation_boxplot.py

Summarize model robustness per transformation in dataset_one_transformation by:
- train accuracy: mean of instances 1-4 per object
- test accuracy: instance 5 per object

Produces two grouped boxplots (train/test) across transformations for all models.
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import transforms

from maps_per_instance_accuracy import (
    DEVICE,
    EXPECTED_NUM_CLASSES,
    LIMIT_PER_INSTANCE,
    MAPS_INDICES_PATH,
    MODEL_SPECS,
    SUPERVISED_TEST_TF,
    TARGET_CLASS,
    TARGET_NAME,
    compute_per_instance_accuracy,
    discover_instance_dirs,
    load_maps_indices,
    resolve_target_class,
)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_PARENT_ROOT = os.path.join(REPO_ROOT, "dataset_one_transformation")
PLOT_ROOT = os.path.join(
    REPO_ROOT,
    "classification_performance_plots",
    "train_test_transformation_boxplot",
)
SHOW_PLOT = False
USE_BAR_PLOT = True
GUESSING_ACCURACY = 1.0 / EXPECTED_NUM_CLASSES
DEFAULT_CACHE_CSV = os.path.join(PLOT_ROOT, "per_instance_accuracy_cache_normalize.csv")
DEFAULT_LOAD_CSV = DEFAULT_CACHE_CSV
DEFAULT_NORMALIZE = True
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
AXIS_LABEL_FONTSIZE = 28
X_TICK_FONTSIZE = 22
Y_TICK_FONTSIZE = 26
FIGSIZE_GROUPED = (20, 8)
FIGSIZE_PER_TRANSFORMATION = (12, 8)
FIGSIZE_OVERALL = (14, 8)
FIGSIZE_CATEGORY = (12, 8)
FIGSIZE_OVERVIEW = (25, 14)
Y_LIMIT_OVERVIEW_TRAIN = 48
Y_LIMIT_OVERVIEW_TEST = 12


def _model_order_key(model_name: str) -> Tuple[int, str]:
    """Sort models as supervised -> ssl -> action, then by name."""
    name = model_name.strip().lower()
    if "supervised" in name:
        group = 0
    elif "ssl" in name:
        group = 1
    elif "action" in name:
        group = 2
    else:
        group = 3
    return group, name


def _model_color(model_name: str) -> str:
    name = model_name.lower()
    if "supervised" in name:
        return "#1f77b4"
    if "ssl" in name:
        return "#ff7f0e"
    if "action" in name:
        return "#2ca02c"
    return "#9467bd"


def _model_category(model_name: str) -> str:
    name = model_name.lower()
    if "supervised" in name:
        return "supervised"
    if "ssl" in name:
        return "ssl"
    if "action" in name:
        return "action"
    return "other"


def _prettify_transformation_label(raw: str) -> str:
    """Convert folder-like names (e.g., background_hue) to title case labels."""
    if raw is None:
        return ""
    label = str(raw).replace("_", " ").replace(".", " ")
    return " ".join(part for part in label.split() if part).title()


def _discover_object_dirs(parent_root: str) -> List[str]:
    if not os.path.isdir(parent_root):
        raise FileNotFoundError(f"Dataset parent root does not exist: {parent_root}")
    object_dirs = [
        os.path.join(parent_root, child)
        for child in sorted(os.listdir(parent_root))
        if os.path.isdir(os.path.join(parent_root, child))
    ]
    if not object_dirs:
        raise RuntimeError(f"No object folders found under: {parent_root}")
    return object_dirs


def _discover_transformations(object_dir: str) -> List[str]:
    candidates = [
        os.path.join(object_dir, child)
        for child in sorted(os.listdir(object_dir))
        if os.path.isdir(os.path.join(object_dir, child))
    ]
    transformations: List[str] = []
    for candidate in candidates:
        try:
            discover_instance_dirs(candidate)
            transformations.append(candidate)
        except Exception:
            continue
    return transformations


def _validate_model_specs() -> List[Dict[str, object]]:
    valid_specs: List[Dict[str, object]] = []
    for spec in MODEL_SPECS:
        model_name = str(spec.get("name", "model")).strip() or "model"
        model_path = os.path.abspath(str(spec.get("path", "")))
        if not model_path or not os.path.isfile(model_path):
            print(f"Skipping '{model_name}': model file not found -> {model_path}")
            continue
        valid_specs.append(spec)
    if not valid_specs:
        raise RuntimeError("No valid model checkpoints found in MODEL_SPECS.")
    return valid_specs


def _compute_counts_from_instances(
    instance_rows: List[Dict[str, object]],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, List[float]]]],
    List[str],
    List[str],
    List[Dict[str, object]],
    Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
]:
    all_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    raw_rows: List[Dict[str, object]] = []
    category_counts: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}

    if not instance_rows:
        return {}, [], [], [], {}

    df = pd.DataFrame(instance_rows)
    if df.empty:
        return {}, [], [], [], {}

    transformations = sorted(df["transformation"].unique().tolist())
    models = sorted(df["model"].unique().tolist(), key=_model_order_key)

    for (trans, model, split), group in df.groupby(["transformation", "model", "split"], as_index=False):
        vals = group["accuracy"].to_numpy(dtype=float).tolist()
        if not vals:
            continue
        all_data.setdefault(trans, {}).setdefault(model, {"train": [], "test": []})
        if split == "train":
            all_data[trans][model]["train"].append(float(np.mean(vals)))
        else:
            all_data[trans][model]["test"].append(float(np.mean(vals)))

    for (obj, trans, model), group in df.groupby(["object", "transformation", "model"], as_index=False):
        train_vals = group[group["split"] == "train"]["accuracy"].to_numpy(dtype=float)
        test_vals = group[group["split"] == "test"]["accuracy"].to_numpy(dtype=float)
        if train_vals.size == 0 or test_vals.size == 0:
            continue
        raw_rows.append(
            {
                "object": obj,
                "transformation": trans,
                "model": model,
                "train_mean": float(np.mean(train_vals)),
                "test_mean": float(np.mean(test_vals)),
            }
        )

    for (trans, model), group in df.groupby(["transformation", "model"], as_index=False):
        category = _model_category(model)
        trans_counts = category_counts.setdefault(trans, {})
        cat_counts = trans_counts.setdefault(category, {
            "train": {"over_guess": 0, "total": 0},
            "test": {"over_guess": 0, "total": 0},
        })
        for split in ("train", "test"):
            split_vals = group[group["split"] == split]["accuracy"].to_numpy(dtype=float)
            cat_counts[split]["over_guess"] += int(np.sum(split_vals > GUESSING_ACCURACY))
            cat_counts[split]["total"] += int(split_vals.size)

    return all_data, transformations, models, raw_rows, category_counts


def _compute_instance_stderr_over_instances(
    instance_rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute std error over all instance accuracies per transformation/model/split."""
    if not instance_rows:
        return {}
    df = pd.DataFrame(instance_rows)
    if df.empty:
        return {}

    stderrs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (trans, model, split), group in df.groupby(["transformation", "model", "split"], as_index=False):
        vals = group["accuracy"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        std = float(np.std(vals, ddof=0))
        stderrs.setdefault(trans, {}).setdefault(model, {})[split] = std / float(np.sqrt(vals.size))
    return stderrs


def _collect_transformation_stats(
    object_dirs: List[str],
    class_to_idx: Dict[str, int],
    model_specs: List[Dict[str, object]],
    apply_normalize: bool = False,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, List[float]]]],
    List[str],
    List[str],
    List[Dict[str, object]],
    Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
    List[Dict[str, object]],
]:
    """Return per-transformation train/test lists for each model."""
    all_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    raw_rows: List[Dict[str, object]] = []
    category_counts: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    instance_rows: List[Dict[str, object]] = []

    all_models = [str(spec.get("name", "model")).strip() or "model" for spec in model_specs]

    for object_dir in object_dirs:
        object_name = os.path.basename(os.path.abspath(object_dir))
        target_class = resolve_target_class(
            dataset_root=object_dir,
            class_to_idx=class_to_idx,
            target_class=TARGET_CLASS,
            target_name=TARGET_NAME,
        )
        transformations = _discover_transformations(object_dir)
        if not transformations:
            print(f"Skipping '{object_name}': no transformation folders found")
            continue

        for trans_dir in transformations:
            trans_name = os.path.basename(os.path.abspath(trans_dir))
            for spec in model_specs:
                model_name = str(spec.get("name", "model")).strip() or "model"
                model_path = os.path.abspath(str(spec.get("path", "")))
                use_sup = bool(spec.get("use_sup_lin_projector", False))

                print(
                    f"Evaluating '{model_name}' on object='{object_name}' "
                    f"transform='{trans_name}'"
                )
                results_df, _ = compute_per_instance_accuracy(
                    dataset_root=trans_dir,
                    model_name=model_path,
                    target_class=target_class,
                    device=DEVICE,
                    use_sup_lin_projector=use_sup,
                    limit_per_instance=LIMIT_PER_INSTANCE,
                    model_label=model_name,
                    print_predictions_debug=False,
                    predictions_debug_preview=None,
                    expected_num_classes=EXPECTED_NUM_CLASSES,
                    preprocess_override=(
                        SUPERVISED_TEST_TF
                        if "supervised" in model_name.lower()
                        else (
                            transforms.Compose(
                                [
                                    transforms.ToTensor(),
                                    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
                                ]
                            )
                            if (apply_normalize and ("ssl" in model_name.lower() or "action" in model_name.lower()))
                            else (
                                transforms.ToTensor()
                                if ("ssl" in model_name.lower() or "action" in model_name.lower())
                                else None
                            )
                        )
                    ),
                )

                category = _model_category(model_name)
                train_acc = results_df.iloc[:4]["accuracy"].to_numpy(dtype=float)
                test_acc = results_df.iloc[4:5]["accuracy"].to_numpy(dtype=float)
                trans_counts = category_counts.setdefault(trans_name, {})
                cat_counts = trans_counts.setdefault(category, {
                    "train": {"over_guess": 0, "total": 0},
                    "test": {"over_guess": 0, "total": 0},
                })
                cat_counts["train"]["over_guess"] += int(np.sum(train_acc > GUESSING_ACCURACY))
                cat_counts["train"]["total"] += int(train_acc.size)
                cat_counts["test"]["over_guess"] += int(np.sum(test_acc > GUESSING_ACCURACY))
                cat_counts["test"]["total"] += int(test_acc.size)

                for idx, acc in enumerate(train_acc.tolist(), start=1):
                    instance_rows.append(
                        {
                            "object": object_name,
                            "transformation": trans_name,
                            "model": model_name,
                            "split": "train",
                            "instance_idx": idx,
                            "accuracy": float(acc),
                        }
                    )
                for idx, acc in enumerate(test_acc.tolist(), start=5):
                    instance_rows.append(
                        {
                            "object": object_name,
                            "transformation": trans_name,
                            "model": model_name,
                            "split": "test",
                            "instance_idx": idx,
                            "accuracy": float(acc),
                        }
                    )

                if train_acc.size == 0 or test_acc.size == 0:
                    print(
                        f"Skipping '{model_name}' on '{object_name}/{trans_name}': "
                        f"expected 5 instances, got {len(results_df)}"
                    )
                    continue

                train_mean = float(np.mean(train_acc))
                test_mean = float(np.mean(test_acc))

                all_data.setdefault(trans_name, {}).setdefault(
                    model_name, {"train": [], "test": []}
                )
                all_data[trans_name][model_name]["train"].append(train_mean)
                all_data[trans_name][model_name]["test"].append(test_mean)

                raw_rows.append(
                    {
                        "object": object_name,
                        "transformation": trans_name,
                        "model": model_name,
                        "train_mean": train_mean,
                        "test_mean": test_mean,
                    }
                )

    transformations_sorted = sorted(all_data.keys())
    models_sorted = sorted(all_models, key=_model_order_key)
    return all_data, transformations_sorted, models_sorted, raw_rows, category_counts, instance_rows


def _plot_grouped_boxplots(
    all_data: Dict[str, Dict[str, Dict[str, List[float]]]],
    transformations: List[str],
    models: List[str],
    plot_root: str,
    show_plot: bool = False,
    use_bar_plot: bool = False,
    instance_stderr: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> None:
    os.makedirs(plot_root, exist_ok=True)
    x_positions = np.arange(len(transformations))
    box_width = 0.8 / max(len(models), 1)

    for split, ylabel, out_name in (
        ("train", "Train Accuracy", "all_transformations_train.png"),
        ("test", "Test Accuracy", "all_transformations_test.png"),
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE_GROUPED)
        data_by_model: Dict[str, List[List[float]]] = {model: [] for model in models}
        pos_by_model: Dict[str, List[float]] = {model: [] for model in models}
        instance_stderr_by_model: Dict[str, List[float]] = {model: [] for model in models}

        for trans_idx, trans in enumerate(transformations):
            for model_idx, model in enumerate(models):
                vals = all_data.get(trans, {}).get(model, {}).get(split, [])
                if not vals:
                    continue
                data_by_model[model].append(vals)
                pos_by_model[model].append(
                    x_positions[trans_idx]
                    + (model_idx - len(models) / 2) * box_width
                    + box_width / 2
                )
                stderr_val = 0.0
                if instance_stderr is not None:
                    stderr_val = float(instance_stderr.get(trans, {}).get(model, {}).get(split, 0.0))
                instance_stderr_by_model[model].append(stderr_val)

        for model in models:
            if not data_by_model[model]:
                continue
            color = _model_color(model)
            if use_bar_plot:
                means = [float(np.mean(v)) for v in data_by_model[model]]
                stds = [float(np.std(v, ddof=0)) for v in data_by_model[model]]
                ax.bar(
                    pos_by_model[model],
                    means,
                    yerr=stds,
                    width=box_width * 0.8,
                    color=color,
                    alpha=0.7,
                    edgecolor=color,
                    linewidth=0.8,
                    label=model,
                    capsize=3,
                )
                ax.errorbar(
                    pos_by_model[model],
                    means,
                    yerr=instance_stderr_by_model[model],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=3,
                    zorder=3,
                )
            else:
                bp = ax.boxplot(
                    data_by_model[model],
                    positions=pos_by_model[model],
                    widths=box_width * 0.8,
                    patch_artist=True,
                    showmeans=True,
                    meanline=False,
                    label=model,
                    flierprops={
                        "marker": "o",
                        "markerfacecolor": color,
                        "markeredgecolor": color,
                        "markersize": 4,
                        "alpha": 0.9,
                    },
                )
                for box in bp["boxes"]:
                    box.set(facecolor=color, alpha=0.6, edgecolor=color, linewidth=0.8)
                for median in bp["medians"]:
                    median.set(color=color, linewidth=1.2)
                for whisker in bp["whiskers"]:
                    whisker.set(color=color, linewidth=0.8)
                for cap in bp["caps"]:
                    cap.set(color=color, linewidth=0.8)
                for mean in bp["means"]:
                    mean.set(marker="o", markerfacecolor=color, markeredgecolor=color, markersize=4)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [_prettify_transformation_label(t) for t in transformations],
            rotation=45,
            ha="right",
            fontsize=X_TICK_FONTSIZE,
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("Transformation Parameter", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.3)
        # Legend hidden for now.

        fig.tight_layout()
        out_path = os.path.join(plot_root, out_name)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def _plot_per_transformation(
    all_data: Dict[str, Dict[str, Dict[str, List[float]]]],
    transformations: List[str],
    models: List[str],
    plot_root: str,
    show_plot: bool = False,
    use_bar_plot: bool = False,
) -> None:
    for trans in transformations:
        trans_data = all_data.get(trans, {})
        if not trans_data:
            continue

        for split, ylabel, suffix in (
            ("train", "Train Accuracy (instances 1-4)", "train"),
            ("test", "Test Accuracy (instance 5)", "test"),
        ):
            fig, ax = plt.subplots(figsize=FIGSIZE_PER_TRANSFORMATION)
            data_by_model: Dict[str, List[List[float]]] = {model: [] for model in models}
            labels: List[str] = []

            for model in models:
                vals = trans_data.get(model, {}).get(split, [])
                if not vals:
                    continue
                data_by_model[model].append(vals)
                labels.append(model)

            for model_idx, model in enumerate(models):
                vals_list = data_by_model[model]
                if not vals_list:
                    continue
                color = _model_color(model)
                if use_bar_plot:
                    means = [float(np.mean(v)) for v in vals_list]
                    stds = [float(np.std(v, ddof=0)) for v in vals_list]
                    ns = [max(len(v), 1) for v in vals_list]
                    stderrs = [stds[idx] / float(np.sqrt(ns[idx])) for idx in range(len(stds))]
                    ax.bar(
                        [model_idx + 1],
                        means,
                        yerr=stderrs,
                        width=0.6,
                        color=color,
                        alpha=0.7,
                        edgecolor=color,
                        linewidth=0.8,
                        capsize=3,
                    )
                else:
                    bp = ax.boxplot(
                        vals_list,
                        positions=[model_idx + 1],
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanline=False,
                        labels=[model],
                        flierprops={
                            "marker": "o",
                            "markerfacecolor": color,
                            "markeredgecolor": color,
                            "markersize": 4,
                            "alpha": 0.9,
                        },
                    )
                    for box in bp["boxes"]:
                        box.set(facecolor=color, alpha=0.6, edgecolor=color, linewidth=0.8)
                    for median in bp["medians"]:
                        median.set(color=color, linewidth=1.2)
                    for whisker in bp["whiskers"]:
                        whisker.set(color=color, linewidth=0.8)
                    for cap in bp["caps"]:
                        cap.set(color=color, linewidth=0.8)
                    for mean in bp["means"]:
                        mean.set(marker="o", markerfacecolor=color, markeredgecolor=color, markersize=4)

            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=X_TICK_FONTSIZE)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_xlabel("Model", fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            out_path = os.path.join(plot_root, f"{trans}_{suffix}.png")
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot: {out_path}")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)


def _plot_overall_mean_over_transformations(
    raw_rows: List[Dict[str, object]],
    models: List[str],
    plot_root: str,
    show_plot: bool = False,
    use_bar_plot: bool = False,
) -> None:
    if not raw_rows:
        print("No raw rows available for overall overview plot.")
        return

    df = pd.DataFrame(raw_rows)
    if df.empty:
        print("No data available for overall overview plot.")
        return

    # Mean over transformations per object, then aggregate across objects per model.
    grouped = df.groupby(["model", "object"], as_index=False).agg(
        train_mean=("train_mean", "mean"),
        test_mean=("test_mean", "mean"),
    )

    for split, ylabel, out_name in (
        ("train_mean", "Mean Train Accuracy (avg over transformations)", "overall_mean_over_transformations_train.png"),
        ("test_mean", "Mean Test Accuracy (avg over transformations)", "overall_mean_over_transformations_test.png"),
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE_OVERALL)
        data_by_model: List[List[float]] = []
        labels: List[str] = []

        for model in models:
            vals = grouped[grouped["model"] == model][split].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            data_by_model.append(vals.tolist())
            labels.append(model)

        if not data_by_model:
            print(f"No data available for {split} overview plot.")
            plt.close(fig)
            continue

        x_positions = np.arange(1, len(labels) + 1)

        if use_bar_plot:
            means = [float(np.mean(v)) for v in data_by_model]
            stds = [float(np.std(v, ddof=0)) for v in data_by_model]
            ns = [max(len(v), 1) for v in data_by_model]
            stderrs = [stds[idx] / float(np.sqrt(ns[idx])) for idx in range(len(stds))]
            colors = [_model_color(m) for m in labels]
            ax.bar(
                x_positions,
                means,
                yerr=stderrs,
                width=0.6,
                color=colors,
                alpha=0.7,
                edgecolor=colors,
                linewidth=0.8,
                capsize=3,
            )
        else:
            bp = ax.boxplot(
                data_by_model,
                positions=x_positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanline=False,
            )
            for idx, box in enumerate(bp["boxes"]):
                color = _model_color(labels[idx])
                box.set(facecolor=color, alpha=0.6, edgecolor=color, linewidth=0.8)
            for idx, median in enumerate(bp["medians"]):
                color = _model_color(labels[idx])
                median.set(color=color, linewidth=1.2)
            for idx, whisker in enumerate(bp["whiskers"]):
                color = _model_color(labels[idx // 2])
                whisker.set(color=color, linewidth=0.8)
            for idx, cap in enumerate(bp["caps"]):
                color = _model_color(labels[idx // 2])
                cap.set(color=color, linewidth=0.8)
            for idx, mean in enumerate(bp["means"]):
                color = _model_color(labels[idx])
                mean.set(marker="o", markerfacecolor=color, markeredgecolor=color, markersize=4)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=X_TICK_FONTSIZE)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("Model", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        os.makedirs(plot_root, exist_ok=True)
        out_path = os.path.join(plot_root, out_name)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def _plot_over_guessing_accuracy_by_category_per_transformation(
    category_counts: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
    transformations: List[str],
    plot_root: str,
    show_plot: bool = False,
) -> None:
    if not category_counts:
        print("No category counts available for guessing-accuracy plots.")
        return

    for trans in transformations:
        trans_counts = category_counts.get(trans, {})
        if not trans_counts:
            continue

        categories = sorted(trans_counts.keys())
        for split in ("train", "test"):
            over_guess = [int(trans_counts[c][split]["over_guess"]) for c in categories]
            totals = [int(trans_counts[c][split]["total"]) for c in categories]
            colors = [_model_color(c) for c in categories]

            fig, ax = plt.subplots(figsize=FIGSIZE_CATEGORY)
            x_positions = np.arange(len(categories))
            bars = ax.bar(
                x_positions,
                over_guess,
                width=0.6,
                color=colors,
                alpha=0.75,
                edgecolor=colors,
                linewidth=0.8,
            )

            for idx, bar in enumerate(bars):
                total = totals[idx]
                label = f"{over_guess[idx]}/{total}" if total > 0 else "0/0"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(1, bar.get_height() * 0.02),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(categories, rotation=0, ha="center", fontsize=X_TICK_FONTSIZE)
            ax.set_ylabel(f"# Instances > guessing accuracy", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_xlabel("Model Category", fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            os.makedirs(plot_root, exist_ok=True)
            out_path = os.path.join(plot_root, f"{trans}_instances_over_guessing_{split}.png")
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot: {out_path}")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)


def _plot_over_guessing_accuracy_overview_by_transformation(
    category_counts: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
    transformations: List[str],
    plot_root: str,
    show_plot: bool = False,
) -> None:
    if not category_counts:
        print("No category counts available for overview guessing-accuracy plot.")
        return

    categories = ["supervised", "ssl", "action"]
    colors = [_model_color(c) for c in categories]

    for split in ("train", "test"):
        fig, ax = plt.subplots(figsize=FIGSIZE_OVERVIEW)
        x_positions = np.arange(len(transformations))
        bar_width = 0.22

        for idx, category in enumerate(categories):
            counts: List[int] = []
            for trans in transformations:
                trans_counts = category_counts.get(trans, {})
                cat_counts = trans_counts.get(category, {
                    "train": {"over_guess": 0, "total": 0},
                    "test": {"over_guess": 0, "total": 0},
                })
                count = int(cat_counts[split]["over_guess"])
                counts.append(count)

            ax.bar(
                x_positions + (idx - 1) * bar_width,
                counts,
                width=bar_width,
                color=colors[idx],
                alpha=0.8,
                edgecolor=colors[idx],
                linewidth=0.8,
                label=category,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [_prettify_transformation_label(t) for t in transformations],
            rotation=45,
            ha="right",
            fontsize=X_TICK_FONTSIZE,
        )
        ax.set_xlim(-0.5, len(transformations) - 0.5)
        ax.set_ylabel(f"# Instances > guessing accuracy", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel("Transformation Parameter", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(0.0, Y_LIMIT_OVERVIEW_TRAIN if split == "train" else Y_LIMIT_OVERVIEW_TEST)
        ax.set_yticks(
            _build_limit_ticks(
                Y_LIMIT_OVERVIEW_TRAIN if split == "train" else Y_LIMIT_OVERVIEW_TEST,
                step=10.0 if split == "train" else 5.0,
            )
        )
        ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
        ax.grid(axis="y", alpha=0.3)
        # Legend hidden for now.

        fig.tight_layout()
        os.makedirs(plot_root, exist_ok=True)
        out_path = os.path.join(plot_root, f"overview_instances_over_guessing_by_transformation_{split}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def _write_summary_csvs(
    all_data: Dict[str, Dict[str, Dict[str, List[float]]]],
    transformations: List[str],
    models: List[str],
    plot_root: str,
    raw_rows: List[Dict[str, object]],
) -> None:
    rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []

    for trans in transformations:
        best_model = None
        best_test_mean = -1.0
        for model in models:
            train_vals = all_data.get(trans, {}).get(model, {}).get("train", [])
            test_vals = all_data.get(trans, {}).get(model, {}).get("test", [])
            if not train_vals and not test_vals:
                continue
            train_mean = float(np.mean(train_vals)) if train_vals else float("nan")
            train_std = float(np.std(train_vals, ddof=0)) if train_vals else float("nan")
            test_mean = float(np.mean(test_vals)) if test_vals else float("nan")
            test_std = float(np.std(test_vals, ddof=0)) if test_vals else float("nan")
            n_objects = int(max(len(train_vals), len(test_vals)))

            rows.append(
                {
                    "transformation": trans,
                    "model": model,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "test_mean": test_mean,
                    "test_std": test_std,
                    "n_objects": n_objects,
                }
            )

            if not np.isnan(test_mean) and test_mean > best_test_mean:
                best_test_mean = test_mean
                best_model = model

        best_rows.append(
            {
                "transformation": trans,
                "best_model_by_test_mean": best_model or "",
                "best_test_mean": best_test_mean,
            }
        )

    raw_path = os.path.join(plot_root, "all_transformations_per_object_accuracy.csv")
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    print(f"Saved CSV: {raw_path}")

    summary_path = os.path.join(plot_root, "all_transformations_train_vs_test_models.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved CSV: {summary_path}")

    best_path = os.path.join(plot_root, "best_model_by_transformation.csv")
    pd.DataFrame(best_rows).to_csv(best_path, index=False)
    print(f"Saved CSV: {best_path}")


def _write_instance_accuracy_csv(
    instance_rows: List[Dict[str, object]],
    out_path: str,
) -> None:
    if not instance_rows:
        print("No instance rows available to write cache CSV.")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(instance_rows).to_csv(out_path, index=False)
    print(f"Saved CSV: {out_path}")


def _load_instance_accuracy_csv(
    csv_path: str,
) -> List[Dict[str, object]]:
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV cache not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"object", "transformation", "model", "split", "instance_idx", "accuracy"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"CSV cache missing columns: {sorted(missing)}")
    return df.to_dict(orient="records")


def _build_limit_ticks(limit: float, step: float = 5.0) -> List[float]:
    ticks = list(np.arange(0, limit + 0.1, step))
    if ticks and ticks[-1] != limit:
        ticks.append(limit)
    elif not ticks:
        ticks = [0.0, limit]
    return ticks


def main() -> None:
    datasets_parent_root = os.path.abspath(DATASETS_PARENT_ROOT)
    maps_indices_path = os.path.abspath(MAPS_INDICES_PATH)

    parser = argparse.ArgumentParser(description="Generate transformation robustness plots")
    parser.add_argument(
        "--bar",
        dest="use_bar",
        action="store_true",
        help="Plot mean accuracy bars with std error instead of boxplots",
    )
    parser.add_argument(
        "--box",
        dest="use_bar",
        action="store_false",
        help="Plot boxplots (default)",
    )
    parser.add_argument(
        "--load-csv",
        dest="load_csv",
        default=DEFAULT_LOAD_CSV,
        help=f"Load cached per-instance accuracies from CSV (default: {DEFAULT_LOAD_CSV})",
    )
    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        default=DEFAULT_CACHE_CSV,
        help=f"Write per-instance accuracies to CSV (default: {DEFAULT_CACHE_CSV})",
    )
    parser.add_argument(
        "--normalize",
        dest="apply_normalize",
        action="store_true",
        help="Apply normalization before evaluation (ImageNet mean/std)",
    )
    parser.set_defaults(use_bar=USE_BAR_PLOT, apply_normalize=DEFAULT_NORMALIZE)
    args = parser.parse_args()

    class_to_idx = load_maps_indices(maps_indices_path)
    object_dirs = _discover_object_dirs(datasets_parent_root)
    valid_model_specs = _validate_model_specs()

    print(f"Found {len(object_dirs)} objects under: {datasets_parent_root}")
    print(f"Using MAPS indices from: {maps_indices_path}")

    if args.load_csv:
        load_path = os.path.abspath(args.load_csv)
        if os.path.isfile(load_path):
            instance_rows = _load_instance_accuracy_csv(load_path)
            all_data, transformations, models, raw_rows, category_counts = _compute_counts_from_instances(
                instance_rows
            )
        else:
            print(f"CSV cache not found, recomputing: {load_path}")
            all_data, transformations, models, raw_rows, category_counts, instance_rows = _collect_transformation_stats(
                object_dirs=object_dirs,
                class_to_idx=class_to_idx,
                model_specs=valid_model_specs,
                apply_normalize=bool(args.apply_normalize),
            )
            if args.save_csv is not None:
                out_path = os.path.abspath(args.save_csv) if args.save_csv else DEFAULT_CACHE_CSV
                _write_instance_accuracy_csv(instance_rows, out_path)
    else:
        all_data, transformations, models, raw_rows, category_counts, instance_rows = _collect_transformation_stats(
            object_dirs=object_dirs,
            class_to_idx=class_to_idx,
            model_specs=valid_model_specs,
            apply_normalize=bool(args.apply_normalize),
        )
        if args.save_csv is not None:
            out_path = os.path.abspath(args.save_csv) if args.save_csv else DEFAULT_CACHE_CSV
            _write_instance_accuracy_csv(instance_rows, out_path)

    instance_stderr = _compute_instance_stderr_over_instances(instance_rows)

    _plot_grouped_boxplots(
        all_data=all_data,
        transformations=transformations,
        models=models,
        plot_root=PLOT_ROOT,
        show_plot=SHOW_PLOT,
        use_bar_plot=bool(args.use_bar),
        instance_stderr=instance_stderr,
    )

    _plot_per_transformation(
        all_data=all_data,
        transformations=transformations,
        models=models,
        plot_root=PLOT_ROOT,
        show_plot=SHOW_PLOT,
        use_bar_plot=bool(args.use_bar),
    )

    _plot_overall_mean_over_transformations(
        raw_rows=raw_rows,
        models=models,
        plot_root=PLOT_ROOT,
        show_plot=SHOW_PLOT,
        use_bar_plot=bool(args.use_bar),
    )

    _plot_over_guessing_accuracy_by_category_per_transformation(
        category_counts=category_counts,
        transformations=transformations,
        plot_root=PLOT_ROOT,
        show_plot=SHOW_PLOT,
    )

    _plot_over_guessing_accuracy_overview_by_transformation(
        category_counts=category_counts,
        transformations=transformations,
        plot_root=PLOT_ROOT,
        show_plot=SHOW_PLOT,
    )

    _write_summary_csvs(
        all_data=all_data,
        transformations=transformations,
        models=models,
        plot_root=PLOT_ROOT,
        raw_rows=raw_rows,
    )


if __name__ == "__main__":
    main()

