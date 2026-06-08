"""
maps_train_test_model_boxplot.py

Create a single comparison plot per MAPS dataset where:
- each x-position is one model,
- a box shows train performance distribution across instances 1-4,
- a point with errorbar shows test performance from instance 5 (mean +/- std).

Usage:
    1) Optionally edit CONFIG values below.
    2) Run: python maps_train_test_model_boxplot.py
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
PLOT_ROOT = os.path.join(REPO_ROOT, "classification_performance_plots", "train_test_model_boxplot")
SHOW_PLOT = False


def _model_order_key(model_name: str) -> tuple:
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


def _collect_model_stats(
    dataset_root: str,
    target_class: int,
    model_specs: List[Dict[str, object]],
    device: Optional[str] = None,
    limit_per_instance: Optional[int] = None,
    expected_num_classes: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate all configured models and return train/test summary rows."""
    rows: List[Dict[str, object]] = []

    for spec in model_specs:
        model_name = str(spec.get("name", "model")).strip() or "model"
        model_path = os.path.abspath(str(spec.get("path", "")))
        use_sup = bool(spec.get("use_sup_lin_projector", False))

        if not model_path or not os.path.isfile(model_path):
            print(f"Skipping '{model_name}': model file not found -> {model_path}")
            continue

        print(f"Evaluating model '{model_name}' on dataset '{os.path.basename(dataset_root)}'")
        results_df, _ = compute_per_instance_accuracy(
            dataset_root=dataset_root,
            model_name=model_path,
            target_class=target_class,
            device=device,
            use_sup_lin_projector=use_sup,
            limit_per_instance=limit_per_instance,
            model_label=model_name,
            print_predictions_debug=False,
            predictions_debug_preview=None,
            expected_num_classes=expected_num_classes,
            preprocess_override=(SUPERVISED_TEST_TF if "supervised" in model_name.lower() else None),
        )

        if results_df.empty:
            print(f"Skipping '{model_name}': no instance results")
            continue

        # Keep MAPS convention: instances 1-4 train, instance 5 test.
        train_df = results_df.iloc[:4]
        test_df = results_df.iloc[4:5]

        if train_df.empty or test_df.empty:
            print(
                f"Skipping '{model_name}': expected at least 5 instances, "
                f"got {len(results_df)}"
            )
            continue

        test_row = test_df.iloc[0]
        rows.append(
            {
                "model": model_name,
                "train_values": train_df["accuracy"].to_numpy(dtype=float),
                "test_mean": float(test_row["accuracy"]),
                "test_std": float(test_row["std"]) if "std" in test_row else 0.0,
            }
        )

    if not rows:
        raise RuntimeError("No valid model results collected; check MODEL_SPECS and checkpoint paths.")

    return pd.DataFrame(rows)


def plot_train_test_by_category(
    all_data: Dict[str, pd.DataFrame],
    plot_root: str,
    target_class: int,
    show_plot: bool = False,
) -> None:
    """
    Create two separate plots:
    - One for training performance (instances 1-4)
    - One for testing performance (instance 5)
    
    x-axis: object categories (datasets)
    y-axis: top-1 accuracy
    Different colors for each model
    """
    # Define colors for models
    colors = {
        "supervised": "#1f77b4",
        "ssl": "#ff7f0e",
        "action": "#2ca02c",
    }
    default_color = "#9467bd"
    
    # Collect all unique models
    all_models = set()
    for stats_df in all_data.values():
        all_models.update(stats_df["model"].tolist())
    
    all_models = sorted(list(all_models), key=_model_order_key)
    
    # Get colors for each model
    model_colors = {}
    for model in all_models:
        model_lower = model.lower()
        for key, color in colors.items():
            if key in model_lower:
                model_colors[model] = color
                break
        if model not in model_colors:
            model_colors[model] = default_color
    
    category_names = sorted(all_data.keys())
    
    # ============ TRAINING PLOT ============
    fig_train, ax_train = plt.subplots(figsize=(14, 6))
    
    x_positions = np.arange(len(category_names))
    box_width = 0.8 / len(all_models)
    
    train_data_by_model = {model: [] for model in all_models}
    train_positions_by_model = {model: [] for model in all_models}
    
    for cat_idx, category in enumerate(category_names):
        stats_df = all_data[category]
        for model_idx, model in enumerate(all_models):
            model_row = stats_df[stats_df["model"] == model]
            if not model_row.empty:
                train_vals = model_row.iloc[0]["train_values"]
                train_data_by_model[model].append(train_vals)
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
                showmeans=True,
                meanline=False,
                label=model,
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
            for mean in bp["means"]:
                mean.set(marker="o", markerfacecolor=model_colors[model], markeredgecolor=model_colors[model], markersize=4)
    
    ax_train.set_xticks(x_positions)
    ax_train.set_xticklabels(category_names, rotation=45, ha="right")
    ax_train.set_ylim(0.0, 1.05)
    ax_train.set_ylabel("Accuracy", fontsize=20)
    ax_train.set_xlabel("Object Category", fontsize=20)
    ax_train.grid(axis="y", alpha=0.3)
    
    fig_train.tight_layout()
    train_plot_path = os.path.join(plot_root, "all_categories_training.png")
    fig_train.savefig(train_plot_path, dpi=150)
    print(f"Saved training plot: {train_plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig_train)
    
    # ============ TESTING PLOT ============
    fig_test, ax_test = plt.subplots(figsize=(14, 6))
    
    test_data_by_model = {model: [] for model in all_models}
    test_positions_by_model = {model: [] for model in all_models}
    test_std_by_model = {model: [] for model in all_models}
    
    for cat_idx, category in enumerate(category_names):
        stats_df = all_data[category]
        for model_idx, model in enumerate(all_models):
            model_row = stats_df[stats_df["model"] == model]
            if not model_row.empty:
                test_mean = model_row.iloc[0]["test_mean"]
                test_std = model_row.iloc[0]["test_std"]
                # Build pseudo-distribution for boxplot
                lo = float(np.clip(test_mean - test_std, 0.0, 1.0))
                hi = float(np.clip(test_mean + test_std, 0.0, 1.0))
                test_data_by_model[model].append([lo, test_mean, test_mean, test_mean, hi])
                test_positions_by_model[model].append(
                    x_positions[cat_idx] + (model_idx - len(all_models)/2) * box_width + box_width/2
                )
                test_std_by_model[model].append(test_std)
    
    for model in all_models:
        if test_data_by_model[model]:
            bp = ax_test.boxplot(
                test_data_by_model[model],
                positions=test_positions_by_model[model],
                widths=box_width * 0.8,
                patch_artist=True,
                showmeans=False,
                label=model,
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
    
    ax_test.set_xticks(x_positions)
    ax_test.set_xticklabels(category_names, rotation=45, ha="right")
    ax_test.set_ylim(0.0, 1.05)
    ax_test.set_ylabel("Top-1 Accuracy")
    ax_test.set_xlabel("Object Category")
    ax_test.grid(axis="y", alpha=0.3)
    
    fig_test.tight_layout()
    test_plot_path = os.path.join(plot_root, "all_categories_testing.png")
    fig_test.savefig(test_plot_path, dpi=150)
    print(f"Saved testing plot: {test_plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig_test)


def main() -> None:
    datasets_parent_root = os.path.abspath(DATASETS_PARENT_ROOT)
    maps_indices_path = os.path.abspath(MAPS_INDICES_PATH)

    dataset_roots = discover_dataset_roots(datasets_parent_root)
    class_to_idx = load_maps_indices(maps_indices_path)

    os.makedirs(PLOT_ROOT, exist_ok=True)

    print(f"Found {len(dataset_roots)} dataset roots in: {datasets_parent_root}")
    
    # Collect stats for all datasets
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

        stats_df = _collect_model_stats(
            dataset_root=dataset_root,
            target_class=target_class,
            model_specs=MODEL_SPECS,
            device=DEVICE,
            limit_per_instance=LIMIT_PER_INSTANCE,
            expected_num_classes=EXPECTED_NUM_CLASSES,
        )
        stats_df = stats_df.sort_values(by="model", key=lambda s: s.map(_model_order_key)).reset_index(drop=True)

        all_data[dataset_name] = stats_df

        # Also save per-dataset CSV
        out_csv = os.path.join(PLOT_ROOT, f"{dataset_name}_train_vs_test_models.csv")
        csv_rows: List[Dict[str, object]] = []
        for _, row in stats_df.iterrows():
            train_vals = np.asarray(row["train_values"], dtype=float)
            csv_rows.append(
                {
                    "model": row["model"],
                    "train_mean": float(np.mean(train_vals)),
                    "train_std": float(np.std(train_vals, ddof=0)),
                    "test_mean": float(row["test_mean"]),
                    "test_std": float(row["test_std"]),
                }
            )

        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv}")
    
    # Create combined plot with all categories
    print("\n" + "-" * 80)
    print("Creating combined plots for all categories...")
    plot_train_test_by_category(
        all_data=all_data,
        plot_root=PLOT_ROOT,
        target_class=target_class_global or 0,
        show_plot=SHOW_PLOT,
    )



if __name__ == "__main__":
    main()

