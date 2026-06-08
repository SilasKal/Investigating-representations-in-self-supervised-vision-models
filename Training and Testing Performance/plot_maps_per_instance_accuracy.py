"""
maps_per_instance_accuracy.py

Evaluate a model on a MAPS-style dataset root that contains 5 instance folders.
For each instance, compute top-1 accuracy against a given target class and plot
per-instance results.

Usage:
    1) Edit the CONFIG section below.
    2) Run: python maps_per_instance_accuracy.py
"""

import os
import re
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from classification_performance import get_model_and_preprocess


_IDX_RE = re.compile(r"(\d+)(?=\.(jpg|jpeg|png)$)", re.IGNORECASE)


# CONFIG: edit these variables directly
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_PARENT_ROOT = os.path.join(REPO_ROOT, "dataset_new_multiple")
MAPS_INDICES_PATH = os.path.join(REPO_ROOT, "maps_indeces")
GLOBAL_PLOT_DIR = os.path.join(
    REPO_ROOT,
    "classification_performance_plots",
    "detailed_performance_comparison",
    "resnet18",
    "not pretrained",
)

# Run all listed models in one evaluation pass for detailed performance comparison (resnet18 / not pretrained).
MODEL_SPECS = [
    {
        "name": "ssl_resnet18_not_pretrained",
        "path": os.path.join(
            REPO_ROOT,
            "detailed_performance_comparison",
            # "resnet18",
            "resnet50",
            "not pretrained",
            "ssl_epoch_49.pt",
        ),
        "use_sup_lin_projector": True,
    },
    {
        "name": "action_resnet18_not_pretrained",
        "path": os.path.join(
            REPO_ROOT,
            "detailed_performance_comparison",
            # "resnet18",
            "resnet50",
            "not pretrained",
            "action_epoch_49.pt",
        ),
        "use_sup_lin_projector": True,
    },
    {
        "name": "supervised_resnet18_not_pretrained",
        "path": os.path.join(
            REPO_ROOT,
            "detailed_performance_comparison",
            # "resnet18",
            "resnet50",
            "not pretrained",
            "supervised_MAPS_resnet50_seed0_1.pt"
            # "supervised_MAPS_resnet18_seed0_1.pt",
        ),
        "use_sup_lin_projector": False,
    },
]

# None/"" => auto: <GLOBAL_PLOT_DIR>/<dataset_name>
OUTPUT_DIR_OVERRIDE = None

# Target class resolution priority: TARGET_CLASS -> TARGET_NAME -> infer from dataset folder name
TARGET_CLASS = None
TARGET_NAME = None

DEVICE = None
LIMIT_PER_INSTANCE = None
SHOW_PLOT = False
PRINT_PREDICTIONS_DEBUG = True
PREDICTIONS_DEBUG_PREVIEW = None  # None => print all predictions, int => print first N
EXPECTED_NUM_CLASSES = 12

SUPERVISED_TEST_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_maps_indices(maps_indices_path: str) -> Dict[str, int]:
    """Load class-name to class-index mapping from MAPS index file."""
    if not os.path.isfile(maps_indices_path):
        raise FileNotFoundError(f"MAPS indices file not found: {maps_indices_path}")

    class_to_idx: Dict[str, int] = {}
    with open(maps_indices_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",", maxsplit=1)]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line in {maps_indices_path} at line {line_num}: '{raw_line.rstrip()}'"
                )

            idx_str, label = parts
            class_to_idx[label] = int(idx_str)

    if not class_to_idx:
        raise RuntimeError(f"No class mappings parsed from: {maps_indices_path}")
    return class_to_idx


def resolve_target_class(
    dataset_root: str,
    class_to_idx: Dict[str, int],
    target_class: Optional[int] = None,
    target_name: Optional[str] = None,
) -> int:
    """Resolve target class index from explicit idx/name or dataset folder name."""
    if target_class is not None:
        return int(target_class)

    if target_name:
        if target_name in class_to_idx:
            return class_to_idx[target_name]

        target_norm = _norm_label(target_name)
        for label, idx in class_to_idx.items():
            if _norm_label(label) == target_norm:
                return idx
        raise ValueError(f"Unknown target_name '{target_name}' in MAPS indices")

    dataset_name = os.path.basename(os.path.abspath(dataset_root)).lower()
    dataset_norm = _norm_label(dataset_name)
    matches: List[Tuple[int, str, int]] = []
    for label, idx in class_to_idx.items():
        label_norm = _norm_label(label)
        if label_norm and label_norm in dataset_norm:
            matches.append((len(label_norm), label, idx))

    if not matches:
        raise ValueError(
            "Could not infer target class from dataset name. "
            "Set TARGET_CLASS or TARGET_NAME in the CONFIG section."
        )

    # Use the longest matching label to avoid ambiguous partial matches.
    matches.sort(reverse=True)
    return matches[0][2]


def _idx_from_name(path: str) -> int:
    """Sort image paths by trailing integer before extension when available."""
    match = _IDX_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else 10 ** 18


def _instance_sort_key(path: str) -> Tuple[int, str]:
    """Sort instance folders by trailing integer in folder name, then by name."""
    name = os.path.basename(path.rstrip(os.sep))
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10 ** 18, name)


def discover_instance_dirs(dataset_root: str) -> List[str]:
    """
    Return immediate subdirectories that look like MAPS instances.
    A valid instance folder must contain an 'images' subfolder.
    """
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    candidates = [
        os.path.join(dataset_root, child)
        for child in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, child))
    ]

    instance_dirs = [
        path for path in candidates
        if os.path.isdir(os.path.join(path, "images"))
    ]
    instance_dirs.sort(key=_instance_sort_key)

    if not instance_dirs:
        raise RuntimeError(
            "No instance folders found. Expected subfolders with an 'images' directory."
        )

    return instance_dirs


def discover_dataset_roots(parent_root: str) -> List[str]:
    """
    Return all immediate subfolders that look like valid MAPS dataset roots.
    A valid root is any folder where discover_instance_dirs() succeeds.
    """
    if not os.path.isdir(parent_root):
        raise FileNotFoundError(f"Datasets parent root does not exist: {parent_root}")

    children = [
        os.path.join(parent_root, child)
        for child in sorted(os.listdir(parent_root))
        if os.path.isdir(os.path.join(parent_root, child))
    ]

    dataset_roots: List[str] = []
    for child in children:
        try:
            discover_instance_dirs(child)
            dataset_roots.append(child)
        except Exception:
            # Skip folders that are not MAPS dataset roots.
            continue

    if not dataset_roots:
        raise RuntimeError(
            "No valid dataset roots found under parent folder. "
            "Expected subfolders that contain instance dirs with images/."
        )

    return dataset_roots


@torch.no_grad()
def infer_instance(
    images_dir: str,
    model,
    preprocess,
    device: str,
    limit: Optional[int] = None,
    expected_num_classes: Optional[int] = None,
    patterns: Sequence[str] = ("*.jpg", "*.jpeg", "*.png"),
) -> np.ndarray:
    """Run model inference for all images in one instance folder."""
    image_paths: List[str] = []
    for pattern in patterns:
        image_paths.extend(glob(os.path.join(images_dir, pattern)))
    image_paths.sort(key=_idx_from_name)

    if not image_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    if limit is not None:
        image_paths = image_paths[: max(int(limit), 0)]

    preds = np.empty(len(image_paths), dtype=int)
    checked_logits_shape = False
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        x = preprocess(image).unsqueeze(0).to(device)
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]

        if expected_num_classes is not None and not checked_logits_shape:
            if y.ndim != 2 or y.shape[1] != int(expected_num_classes):
                raise RuntimeError(
                    "Model output does not match expected class logits. "
                    f"Expected shape [N, {expected_num_classes}], got {tuple(y.shape)}. "
                    "For SSL/action checkpoints, enable the classification head "
                    "(use_sup_lin_projector=True)."
                )
            checked_logits_shape = True

        preds[i] = int(torch.argmax(y, dim=1).item())

    return preds


def compute_per_instance_accuracy(
    dataset_root: str,
    model_name: str,
    target_class: int,
    device: Optional[str] = None,
    use_sup_lin_projector: bool = False,
    limit_per_instance: Optional[int] = None,
    model_label: Optional[str] = None,
    print_predictions_debug: bool = False,
    predictions_debug_preview: Optional[int] = None,
    expected_num_classes: Optional[int] = None,
    preprocess_override=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute top-1 accuracy per MAPS instance directory.

    Returns
    -------
    results_df : DataFrame with columns instance, n_images, n_correct, accuracy, std
    debug_df   : DataFrame with columns instance, image_name, pred_class
    """
    model, preprocess, model_device = get_model_and_preprocess(
        device=device,
        model_name=model_name,
        use_sup_lin_projector=use_sup_lin_projector,
    )
    if preprocess_override is not None:
        preprocess = preprocess_override

    instance_dirs = discover_instance_dirs(dataset_root)
    if len(instance_dirs) != 5:
        print(f"Warning: expected 5 instances, found {len(instance_dirs)}")

    rows = []
    debug_rows = []
    for instance_dir in instance_dirs:
        images_dir = os.path.join(instance_dir, "images")
        instance_name = os.path.basename(instance_dir)
        image_paths: List[str] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(glob(os.path.join(images_dir, pattern)))
        image_paths.sort(key=_idx_from_name)

        preds = infer_instance(
            images_dir=images_dir,
            model=model,
            preprocess=preprocess,
            device=model_device,
            limit=limit_per_instance,
            expected_num_classes=expected_num_classes,
        )

        correct = int(np.sum(preds == target_class))
        n_images = int(len(preds))
        acc = float(correct / n_images) if n_images > 0 else float("nan")
        std = float(np.sqrt((acc * (1.0 - acc)) / n_images)) if n_images > 0 else float("nan")

        rows.append(
            {
                "instance": instance_name,
                "n_images": n_images,
                "n_correct": correct,
                "accuracy": acc,
                "std": std,
            }
        )

        if len(image_paths) == len(preds):
            for img_path, pred in zip(image_paths, preds):
                debug_rows.append(
                    {
                        "instance": instance_name,
                        "image_name": os.path.basename(img_path),
                        "pred_class": int(pred),
                    }
                )

        print(
            f"[{instance_name}] images={n_images} correct={correct} "
            f"acc={acc * 100:.2f}%"
        )

        if print_predictions_debug:
            preview_preds = preds
            if predictions_debug_preview is not None:
                preview_preds = preds[: max(int(predictions_debug_preview), 0)]
            prefix = f"[{model_label}] " if model_label else ""
            print(f"{prefix}{instance_name} preds: {preview_preds.tolist()}")

    results_df = pd.DataFrame(rows)
    debug_df = pd.DataFrame(debug_rows)
    return results_df, debug_df


def plot_per_instance_accuracy(
    results_df: pd.DataFrame,
    out_png: str,
    model_label: str,
    target_class: int,
    dataset_root: str,
    show_plot: bool = False,
) -> None:
    """Create and save a per-instance bar chart."""
    x_labels = results_df["instance"].tolist()
    y_vals = results_df["accuracy"].to_numpy(dtype=float)
    y_err = results_df["std"].to_numpy(dtype=float) if "std" in results_df.columns else np.zeros_like(y_vals)
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(
        x,
        y_vals,
        yerr=y_err,
        capsize=4,
        color="#4472c4",
        alpha=0.85,
        ecolor="#1f1f1f",
        error_kw={"elinewidth": 1.1},
    )

    train_vals = y_vals[:4] if len(y_vals) >= 4 else y_vals
    train_mean = float(np.nanmean(train_vals))
    train_std = float(np.nanstd(train_vals))
    ax.axhline(
        train_mean,
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
        label=f"train mean (1-4)={train_mean:.3f}",
    )
    if not np.isnan(train_std) and train_std > 0:
        ax.axhspan(
            max(0.0, train_mean - train_std),
            min(1.0, train_mean + train_std),
            color="#d62728",
            alpha=0.10,
            label=f"train std={train_std:.3f}",
        )

    for rect, y, y_std in zip(bars, y_vals, y_err):
        if np.isnan(y):
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y + (0.0 if np.isnan(y_std) else y_std) + 0.02,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ymax = np.nanmax(y_vals + np.nan_to_num(y_err, nan=0.0)) if len(y_vals) else 1.0
    ax.set_ylim(0.0, max(1.05, min(1.2, ymax + 0.08)))
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(
        f"Per-instance MAPS Accuracy model={model_label} target_class={target_class} dataset={os.path.basename(dataset_root)}"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def resolve_output_dir(dataset_root: str, output_dir_override: Optional[str]) -> str:
    """Use manual output path when provided, otherwise save to global plot folder."""
    if output_dir_override and str(output_dir_override).strip():
        return os.path.abspath(str(output_dir_override))

    dataset_name = os.path.basename(os.path.abspath(dataset_root))
    return os.path.join(os.path.abspath(GLOBAL_PLOT_DIR), dataset_name)


def main() -> None:
    datasets_parent_root = os.path.abspath(DATASETS_PARENT_ROOT)
    dataset_roots = discover_dataset_roots(datasets_parent_root)

    maps_indices_path = os.path.abspath(MAPS_INDICES_PATH)
    class_to_idx = load_maps_indices(maps_indices_path)

    print("Running with CONFIG values from maps_per_instance_accuracy.py")
    print(f"Discovered {len(dataset_roots)} dataset roots under: {datasets_parent_root}")
    print(f"Using MAPS indices from: {maps_indices_path}")

    if not MODEL_SPECS:
        raise ValueError("MODEL_SPECS is empty. Add at least one model in the CONFIG section.")

    for dataset_root in dataset_roots:
        dataset_root = os.path.abspath(dataset_root)
        output_dir = resolve_output_dir(dataset_root, OUTPUT_DIR_OVERRIDE)
        os.makedirs(output_dir, exist_ok=True)

        target_class = resolve_target_class(
            dataset_root=dataset_root,
            class_to_idx=class_to_idx,
            target_class=TARGET_CLASS,
            target_name=TARGET_NAME,
        )
        print("\n" + "-" * 80)
        print(f"Dataset: {dataset_root}")
        print(f"Resolved target_class={target_class}")

        for spec in MODEL_SPECS:
            model_name = str(spec.get("name", "model")).strip() or "model"
            model_path = os.path.abspath(str(spec.get("path", "")))
            use_sup = bool(spec.get("use_sup_lin_projector", False))

            if not model_path:
                print(f"Skipping '{model_name}': empty model path")
                continue
            if not os.path.isfile(model_path):
                print(f"Skipping '{model_name}': model file not found -> {model_path}")
                continue

            print(f"\n=== Evaluating model: {model_name} ===")
            print(f"Checkpoint: {model_path}")

            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            results_df, debug_df = compute_per_instance_accuracy(
                dataset_root=dataset_root,
                model_name=model_path,
                target_class=target_class,
                device=DEVICE,
                use_sup_lin_projector=use_sup,
                limit_per_instance=LIMIT_PER_INSTANCE,
                model_label=model_name,
                print_predictions_debug=bool(PRINT_PREDICTIONS_DEBUG),
                predictions_debug_preview=PREDICTIONS_DEBUG_PREVIEW,
                expected_num_classes=EXPECTED_NUM_CLASSES,
                preprocess_override=(SUPERVISED_TEST_TF if "supervised" in model_name.lower() else None),
            )

            csv_path = os.path.join(model_output_dir, "per_instance_accuracy.csv")
            png_path = os.path.join(model_output_dir, "per_instance_accuracy.png")
            debug_csv_path = os.path.join(model_output_dir, "predictions_debug.csv")

            results_df.to_csv(csv_path, index=False)
            debug_df.to_csv(debug_csv_path, index=False)
            plot_per_instance_accuracy(
                results_df=results_df,
                out_png=png_path,
                model_label=model_name,
                target_class=target_class,
                dataset_root=dataset_root,
                show_plot=bool(SHOW_PLOT),
            )

            print("Saved outputs:")
            print(f"  CSV:         {csv_path}")
            print(f"  Plot:        {png_path}")
            print(f"  Predictions: {debug_csv_path}")


if __name__ == "__main__":
    main()

