"""
plot_accuracy_vs_cosine_similarity.py

Erzeugt zwei Scatter-Plots für MAPS-style Daten aus `dataset_new_multiple`:

1) Train-Split:
   x = Klassifikationsgenauigkeit pro Objekt/Instanz
   y = mittlere Cosine Similarity zwischen benachbarten Bildern in der Instanz

2) Test-Split:
   gleiche Struktur, aber nur für die letzte Instanz pro Objekt

Die Punkte werden pro Objekt zusammengefasst, damit die beiden Scatter-Plots
leicht lesbar bleiben. Zusätzlich wird eine CSV mit den Instanz-Metriken und
eine CSV mit den aggregierten Objekt-Summaries gespeichert.

Usage:
    python plot_accuracy_vs_cosine_similarity.py

Optional:
    python plot_accuracy_vs_cosine_similarity.py --objects banana strawberry
    python plot_accuracy_vs_cosine_similarity.py --limit-per-instance 100
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from classification_performance import get_model_and_preprocess


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_DIR / "dataset_new_multiple"
DEFAULT_MAPS_INDICES = PROJECT_DIR / "maps_indeces"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "accuracy_similarity_plots"
DEFAULT_BATCH_SIZE = 64
IDX_RE = re.compile(r"(\d+)(?=\.(jpg|jpeg|png)$)", re.IGNORECASE)
DEFAULT_LABEL_FONTSIZE = 20
DEFAULT_TICK_FONTSIZE = 10
DEFAULT_INSTANCE_CSV_NAME = "instance_metrics.csv"
DEFAULT_USE_CACHED_METRICS = True


DEFAULT_MODEL_SPECS = [
    {
        "name": "action",
        "path": PROJECT_DIR / "model_files" / "resnet50" / "action" / "epoch_49.pt",
        "use_sup_lin_projector": True,
        "add_imagenet_normalize": True,
    },
    {
        "name": "ssl",
        "path": PROJECT_DIR / "model_files" / "resnet50" / "ssl" / "epoch_49.pt",
        "use_sup_lin_projector": True,
        "add_imagenet_normalize": True,
    },
    {
        "name": "supervised",
        # "path": PROJECT_DIR / "model_files" / "supervised_MAPS_resnet18_seed0_1.pt",
        "path" : PROJECT_DIR / "model_files" / "resnet50" / "supervised" / "supervised_MAPS_resnet50_seed0_1.pt",
        "use_sup_lin_projector": True,
        "add_imagenet_normalize": True,
    },
# C:\Users\silas\PycharmProjects\SimClr_MT\model_files\supervised_MAPS_resnet18_seed0_1.pt
]


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _idx_from_name(path: str) -> int:
    match = IDX_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else 10 ** 18


def load_maps_indices(maps_indices_path: Path) -> Dict[str, int]:
    if not maps_indices_path.is_file():
        raise FileNotFoundError(f"MAPS indices file not found: {maps_indices_path}")

    class_to_idx: Dict[str, int] = {}
    with maps_indices_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",", maxsplit=1)]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line in {maps_indices_path} at line {line_num}: {raw_line.rstrip()}"
                )

            idx_str, label = parts
            class_to_idx[label] = int(idx_str)

    if not class_to_idx:
        raise RuntimeError(f"No class mappings parsed from: {maps_indices_path}")
    return class_to_idx


def resolve_target_class(dataset_name: str, class_to_idx: Dict[str, int]) -> int:
    dataset_norm = _norm_label(dataset_name)
    matches: List[Tuple[int, str, int]] = []
    for label, idx in class_to_idx.items():
        label_norm = _norm_label(label)
        if label_norm and label_norm in dataset_norm:
            matches.append((len(label_norm), label, idx))

    if not matches:
        raise ValueError(
            f"Could not infer target class from dataset name '{dataset_name}'. "
            "Pass a dataset root whose folder name contains the class label from `maps_indeces`."
        )

    matches.sort(reverse=True)
    return matches[0][2]


def discover_object_dirs(dataset_root: Path) -> List[Path]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    object_dirs = [
        dataset_root / child
        for child in sorted(os.listdir(dataset_root))
        if (dataset_root / child).is_dir()
    ]

    if not object_dirs:
        raise RuntimeError(f"No object folders found under: {dataset_root}")
    return object_dirs


def discover_instance_dirs(object_dir: Path) -> List[Path]:
    instance_dirs = [
        object_dir / child
        for child in os.listdir(object_dir)
        if (object_dir / child).is_dir() and (object_dir / child / "images").is_dir()
    ]
    instance_dirs.sort(key=lambda p: (re.search(r"(\d+)$", p.name) is None, _idx_from_name(p.name), p.name))
    return instance_dirs


def discover_image_paths(images_dir: Path, patterns: Sequence[str] = ("*.jpg", "*.jpeg", "*.png")) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path(images_dir).glob(pattern))
    paths = sorted(paths, key=lambda p: _idx_from_name(p.name))
    return paths


def add_imagenet_normalization(preprocess):
    """Append the ImageNet normalization used during supervised MAPS training."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if isinstance(preprocess, T.Compose):
        return T.Compose(list(preprocess.transforms) + [normalize])
    return T.Compose([preprocess, normalize])


@torch.no_grad()
def _forward_batch(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


@torch.no_grad()
def compute_instance_metrics(
    images_dir: Path,
    target_class: int,
    full_model: torch.nn.Module,
    feature_model: torch.nn.Module,
    preprocess,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    image_paths = discover_image_paths(images_dir)
    if limit is not None:
        image_paths = image_paths[: max(int(limit), 0)]

    if not image_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    preds: List[int] = []
    embeddings: List[torch.Tensor] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = torch.stack([
            preprocess(Image.open(path).convert("RGB"))
            for path in batch_paths
        ]).to(device)

        logits = _forward_batch(full_model, batch)
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

        emb = _forward_batch(feature_model, batch)
        if emb.ndim > 2:
            emb = torch.flatten(emb, 1)
        emb = F.normalize(emb, dim=1)
        embeddings.append(emb.detach().cpu())

    pred_arr = np.asarray(preds, dtype=int)
    accuracy = float(np.mean(pred_arr == int(target_class)))

    emb_all = torch.cat(embeddings, dim=0)
    if len(emb_all) >= 2:
        consecutive_similarity = torch.sum(emb_all[:-1] * emb_all[1:], dim=1).numpy()
        mean_similarity = float(np.mean(consecutive_similarity))
        similarity_std = float(np.std(consecutive_similarity, ddof=0))
        n_pairs = int(len(consecutive_similarity))
    else:
        mean_similarity = float("nan")
        similarity_std = float("nan")
        n_pairs = 0

    return {
        "n_images": int(len(image_paths)),
        "n_pairs": n_pairs,
        "accuracy": accuracy,
        "mean_cosine_similarity": mean_similarity,
        "similarity_std": similarity_std,
    }


def build_metrics_table(
    dataset_root: Path,
    class_to_idx: Dict[str, int],
    full_model: torch.nn.Module,
    feature_model: torch.nn.Module,
    preprocess,
    device: str,
    objects_filter: Optional[Sequence[str]] = None,
    max_objects: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit_per_instance: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    object_dirs = discover_object_dirs(dataset_root)

    if objects_filter:
        wanted = {_norm_label(name) for name in objects_filter}
        object_dirs = [path for path in object_dirs if _norm_label(path.name) in wanted]

    if max_objects is not None:
        object_dirs = object_dirs[: max(int(max_objects), 0)]

    if not object_dirs:
        raise RuntimeError("No object directories left after filtering.")

    for object_dir in object_dirs:
        instance_dirs = discover_instance_dirs(object_dir)
        if len(instance_dirs) < 2:
            print(f"Skipping {object_dir.name}: expected at least 2 instance folders, found {len(instance_dirs)}")
            continue

        target_class = resolve_target_class(object_dir.name, class_to_idx)
        train_instance_dirs = instance_dirs[:-1]
        test_instance_dirs = instance_dirs[-1:]

        for split_name, split_instances in (("train", train_instance_dirs), ("test", test_instance_dirs)):
            for instance_dir in split_instances:
                images_dir = instance_dir / "images"
                metrics = compute_instance_metrics(
                    images_dir=images_dir,
                    target_class=target_class,
                    full_model=full_model,
                    feature_model=feature_model,
                    preprocess=preprocess,
                    device=device,
                    batch_size=batch_size,
                    limit=limit_per_instance,
                )
                rows.append(
                    {
                        "object": object_dir.name,
                        "instance": instance_dir.name,
                        "split": split_name,
                        "target_class": target_class,
                        **metrics,
                    }
                )
                print(
                    f"[{object_dir.name} / {instance_dir.name} / {split_name}] "
                    f"acc={metrics['accuracy']:.4f} mean_cos={metrics['mean_cosine_similarity']:.4f} "
                    f"n_images={metrics['n_images']}"
                )

    if not rows:
        raise RuntimeError("No metrics could be computed. Check dataset structure and filters.")

    return pd.DataFrame(rows)


def load_instance_metrics_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Instance metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {
        "object",
        "instance",
        "split",
        "target_class",
        "n_images",
        "n_pairs",
        "accuracy",
        "mean_cosine_similarity",
        "similarity_std",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Instance metrics CSV missing columns: {sorted(missing)}")
    return df


def aggregate_by_object(split_df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        split_df.groupby("object", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            mean_cosine_similarity=("mean_cosine_similarity", "mean"),
            n_instances=("instance", "count"),
            n_images=("n_images", "sum"),
            n_pairs=("n_pairs", "sum"),
        )
        .sort_values("object")
        .reset_index(drop=True)
    )
    return aggregated


def plot_scatter(
    summary_df: pd.DataFrame,
    title: str,
    out_path: Path,
    show_plot: bool = False,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
) -> None:
    if summary_df.empty:
        raise RuntimeError(f"No data to plot for {title!r}")

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(summary_df)))

    for color, (_, row) in zip(colors, summary_df.iterrows()):
        x = float(row["mean_cosine_similarity"])
        y = float(row["accuracy"])
        label = f"{row['object']} (n={int(row['n_instances'])})"
        ax.scatter(x, y, s=70, color=color, edgecolors="white", linewidths=0.8, alpha=0.95)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    x_min = float(np.nanmin(summary_df["mean_cosine_similarity"]))
    x_max = float(np.nanmax(summary_df["mean_cosine_similarity"]))
    y_min = max(0.0, float(np.nanmin(summary_df["accuracy"])) - 0.05)
    y_max = min(1.0, float(np.nanmax(summary_df["accuracy"])) + 0.05)
    x_pad = 0.03 if np.isfinite(x_min) and np.isfinite(x_max) else 0.05
    x_min = max(-1.0, x_min - x_pad)
    x_max = min(1.0, x_max + x_pad)

    if x_min == x_max:
        x_min = max(-1.0, x_min - 0.1)
        x_max = min(1.0, x_max + 0.1)
    if y_min == y_max:
        y_min = max(0.0, y_min - 0.1)
        y_max = min(1.0, y_max + 0.1)

    ax.set_xlabel("Cosine similarity of consecutive images", fontsize=label_fontsize)
    ax.set_ylabel("Accuracy", fontsize=label_fontsize)
    # ax.set_title(title)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.625, 1)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_model_overview_scatter(
    points: Sequence[Dict[str, object]],
    title: str,
    out_path: Path,
    show_plot: bool = False,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
) -> None:
    if not points:
        raise RuntimeError(f"No model summary points available for {title!r}")

    labels = [str(point["label"]) for point in points]
    data = [np.asarray(point["cosine_values"], dtype=float) for point in points]
    colors = [str(point.get("color", "tab:gray")) for point in points]

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel("Model type", fontsize=label_fontsize)
    # Use a two-line label with smaller font so it doesn't get cut off in plots
    ax.set_ylabel("Mean cosine similarity\nof consecutive images", fontsize=label_fontsize)
    # ax.set_title(title)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_box_by_object(
    split_df: pd.DataFrame,
    title: str,
    out_path: Path,
    show_plot: bool = False,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
) -> None:
    if split_df.empty:
        raise RuntimeError(f"No data to plot for {title!r}")

    objects = sorted(split_df["object"].unique())
    data = [
        split_df.loc[split_df["object"] == obj, "mean_cosine_similarity"].dropna().to_numpy()
        for obj in objects
    ]

    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(objects)), 6))
    ax.boxplot(
        data,
        labels=objects,
        showmeans=True,
        meanline=True,
        patch_artist=True,
    )
    ax.set_xlabel("Object category", fontsize=label_fontsize)
    ax.set_ylabel("Cosine similarity of consecutive images", fontsize=label_fontsize)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def resolve_output_dir(base_output_dir: Path, model_name: str) -> Path:
    return base_output_dir / model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter plot of MAPS accuracy vs. cosine similarity for train and test splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root folder containing object subfolders (default: dataset_new_multiple).",
    )
    parser.add_argument(
        "--maps-indices",
        type=str,
        default=str(DEFAULT_MAPS_INDICES),
        help="Path to the MAPS index mapping file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Optional single checkpoint or model path used for both accuracy and embeddings. "
            "If omitted, the script processes the built-in action/ssl/supervised model paths."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Display name for --model-path. Defaults to the checkpoint stem.",
    )
    parser.add_argument(
        "--use-sup-lin-projector",
        action="store_true",
        help="Load the supervised linear projector as classifier head when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots and CSVs are written.",
    )
    parser.add_argument(
        "--objects",
        action="append",
        help="Optional object name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Optional cap on the number of object folders processed after filtering.",
    )
    parser.add_argument(
        "--limit-per-instance",
        type=int,
        default=None,
        help="Optional limit on the number of images loaded per instance.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size used for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device, e.g. cpu, cuda, cuda:0. If omitted, auto-detect.",
    )
    parser.add_argument(
        "--label-fontsize",
        type=int,
        default=DEFAULT_LABEL_FONTSIZE,
        help="Font size for x/y axis labels.",
    )
    parser.add_argument(
        "--tick-fontsize",
        type=int,
        default=DEFAULT_TICK_FONTSIZE,
        help="Font size for x/y tick labels.",
    )
    parser.add_argument(
        "--use-cached-metrics",
        action="store_true",
        default=DEFAULT_USE_CACHED_METRICS,
        help="Reuse cached instance metrics CSVs per model if available.",
    )
    parser.add_argument(
        "--instance-csv-name",
        type=str,
        default=DEFAULT_INSTANCE_CSV_NAME,
        help="Filename for per-model instance metrics CSV cache.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots interactively after saving them.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    maps_indices_path = Path(args.maps_indices).resolve()
    base_output_dir = Path(args.output_dir).resolve()

    class_to_idx = load_maps_indices(maps_indices_path)

    if args.model_path:
        model_name = args.model_name or Path(args.model_path).stem or "model"
        model_specs = [
            {
                "name": model_name,
                "path": Path(args.model_path).resolve(),
                "use_sup_lin_projector": bool(args.use_sup_lin_projector),
            }
        ]
    else:
        model_specs = DEFAULT_MODEL_SPECS

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_overview = {"train": [], "test": []}

    for spec in model_specs:
        model_name = str(spec["name"])
        model_path = Path(spec["path"]).resolve()
        output_dir = resolve_output_dir(base_output_dir, model_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing model: {model_name} ===")
        print(f"Model path: {model_path}")

        full_model, preprocess, resolved_device = get_model_and_preprocess(
            device=device,
            model_name=str(model_path),
            use_sup_lin_projector=bool(spec.get("use_sup_lin_projector", False)),
        )
        if spec.get("add_imagenet_normalize", False):
            preprocess = add_imagenet_normalization(preprocess)
        feature_model, _, _ = get_model_and_preprocess(
            device=device,
            model_name=str(model_path),
            use_sup_lin_projector=bool(spec.get("use_sup_lin_projector", False)),
        )
        if hasattr(feature_model, "fc"):
            feature_model.fc = torch.nn.Identity()
        feature_model.eval()

        instance_csv = output_dir / str(args.instance_csv_name)
        if args.use_cached_metrics and instance_csv.is_file():
            print(f"Loading cached instance metrics CSV: {instance_csv}")
            instance_df = load_instance_metrics_csv(instance_csv)
        else:
            instance_df = build_metrics_table(
                dataset_root=dataset_root,
                class_to_idx=class_to_idx,
                full_model=full_model,
                feature_model=feature_model,
                preprocess=preprocess,
                device=resolved_device,
                objects_filter=args.objects,
                max_objects=args.max_objects,
                batch_size=max(int(args.batch_size), 1),
                limit_per_instance=args.limit_per_instance,
            )
            instance_df.to_csv(instance_csv, index=False)

        color_map = {"action": "tab:green", "ssl": "tab:orange", "supervised": "tab:blue"}
        for split_name in ("train", "test"):
            split_df = instance_df.loc[instance_df["split"] == split_name]
            if split_df.empty:
                print(f"Skipping overview for {model_name} ({split_name}): no rows found.")
                continue
            model_overview[split_name].append(
                {
                    "label": model_name,
                    "cosine_values": split_df["mean_cosine_similarity"].astype(float).to_list(),
                    "color": color_map.get(model_name.lower(), "tab:gray"),
                }
            )

        print(f"Saved instance metrics CSV: {instance_csv}")

    for split_name in ("train", "test"):
        overview_png = base_output_dir / f"{split_name}_model_overview.png"
        overview_title = f"Model overview | {split_name.title()} split: mean cosine similarity by model"
        plot_model_overview_scatter(
            model_overview[split_name],
            title=overview_title,
            out_path=overview_png,
            show_plot=bool(args.show),
            label_fontsize=int(args.label_fontsize),
            tick_fontsize=int(args.tick_fontsize),
        )
        print(f"Saved {split_name} model overview scatter plot: {overview_png}")


if __name__ == "__main__":
    main()

