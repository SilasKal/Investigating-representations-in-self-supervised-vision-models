r"""Plot multi-object PCA over multiple epochs.

This script combines the dataset traversal from `plot_multi_object_pca_by_category.py`
with the checkpoint/epoch iteration from the checkpoint evolution scripts.

What it does:
- loads a hierarchical dataset laid out as:
    dataset_root/
      object_name/
        parameter_name/
          instance_*/
            parameters.csv
            images/
- collects one parameter folder at a time across multiple object categories
- loads a sequence of model checkpoints (action / ssl / supervised)
- extracts embeddings at a chosen layer
- fits one common 3D PCA over all epochs for a shared coordinate system
- creates:
    - an epoch grid (6 subplots with 3D scatter plots)
    - a 3D overlay plot
    - a 3D trajectory plot
    - a CSV file with the 3D coordinates (pc1, pc2, pc3)

Example (PowerShell):
    python plot_multi_object_pca_over_epochs.py `
        --dataset-root "C:\Users\silas\PycharmProjects\SimClr_MT\dataset_one_transformation" `
        --model-root "C:\Users\silas\PycharmProjects\SimClr_MT\model_files" `
        --model-arch resnet18 `
        --model-types action ssl supervised `
        --epochs 9 19 29 39 49 `
        --parameter-dir background_hue
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

try:
    import pacmap
    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False

from pretrained.load_mvimgnet_model import load_mv_model

DEFAULT_DATASET_ROOT = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_one_transformation")
DEFAULT_MODEL_ROOT = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files")
DEFAULT_MODEL_ARCH = "resnet18"
DEFAULT_MODEL_TYPES = ["action", "ssl", "supervised"]
DEFAULT_LAYER = "layer4"
DEFAULT_EPOCHS = [9, 19, 29, 39, 49]
DEFAULT_PARAMETER_DIR = None
DEFAULT_SUPERVISED_MODEL = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\supervised")
DEFAULT_USE_PACMAP = False
DEFAULT_X_TICK_FONTSIZE = 12
DEFAULT_Y_TICK_FONTSIZE = 12
DEFAULT_LEGEND_FONTSIZE = 9


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else math.inf, path.name)


def normalize_param_dir_name(name: str) -> str:
    return name.replace(".", "_").strip()


def collect_parameter_dirs(dataset_root: Path, objects: list[str] | None = None) -> list[str]:
    """Collect parameter folder names from the selected object folders."""
    if objects:
        object_dirs = [dataset_root / obj for obj in objects]
    else:
        object_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]

    params: set[str] = set()
    for object_dir in object_dirs:
        if not object_dir.exists():
            continue
        for param_dir in object_dir.iterdir():
            if param_dir.is_dir():
                params.add(param_dir.name)
    return sorted(params)


class MultiObjectParameterDataset(Dataset):
    """Collect images from multiple objects for a single parameter folder."""

    def __init__(
        self,
        dataset_root: str | Path,
        parameter_dir: str,
        objects: list[str] | None = None,
        filename_col: str = "image",
        csv_name: str = "parameters.csv",
        image_subdir: str = "images",
        instance_glob: str = "*",
        transform=None,
        drop_missing_files: bool = True,
    ) -> None:
        self.root = Path(dataset_root)
        self.parameter_dir = normalize_param_dir_name(parameter_dir)
        self.filename_col = filename_col
        self.transform = transform or T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if objects:
            object_dirs = [self.root / obj for obj in objects]
            missing = [p.name for p in object_dirs if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Requested objects not found: {missing}")
        else:
            object_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])

        records: list[dict[str, object]] = []
        kept_objects: list[str] = []

        for object_dir in object_dirs:
            param_dir = object_dir / self.parameter_dir
            if not param_dir.exists() or not param_dir.is_dir():
                continue

            instance_dirs = sorted(
                [p for p in param_dir.glob(instance_glob) if p.is_dir() and (p / csv_name).exists()],
                key=natural_key,
            )
            if not instance_dirs:
                continue

            kept_objects.append(object_dir.name)
            for instance_dir in instance_dirs:
                csv_path = instance_dir / csv_name
                img_dir = instance_dir / image_subdir
                if not img_dir.exists():
                    continue

                df = pd.read_csv(csv_path)
                if filename_col not in df.columns:
                    raise ValueError(f"Missing filename column '{filename_col}' in {csv_path}")

                for local_idx, filename in enumerate(df[filename_col].astype(str)):
                    img_path = img_dir / filename
                    if drop_missing_files and not img_path.exists():
                        continue
                    records.append(
                        {
                            "object_name": object_dir.name,
                            "instance_name": instance_dir.name,
                            "local_idx": int(local_idx),
                            "path": img_path,
                            "filename": img_path.name,
                            "relative_path": str(img_path.relative_to(self.root)),
                        }
                    )

        if not records:
            raise RuntimeError(
                f"No samples found for parameter '{self.parameter_dir}' in dataset root {self.root}"
            )

        self.df = pd.DataFrame(records)
        self.paths = [Path(p) for p in self.df["path"].tolist()]
        self.object_names = self.df["object_name"].astype(str).tolist()
        self.instance_names = self.df["instance_name"].astype(str).tolist()
        self.filenames = self.df["filename"].astype(str).tolist()
        self.relative_paths = self.df["relative_path"].astype(str).tolist()

        self.object_labels = sorted(set(self.object_names))
        self.object_to_idx = {name: i for i, name in enumerate(self.object_labels)}
        self.object_indices = np.array([self.object_to_idx[name] for name in self.object_names], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        object_idx = torch.tensor(self.object_indices[idx], dtype=torch.long)
        return image, object_idx, idx


def resolve_checkpoint_paths(
    checkpoint_paths: Sequence[str | Path] | None,
    checkpoint_dir: str | Path | None,
    checkpoint_pattern: str,
) -> list[Path]:
    if checkpoint_paths:
        paths = [Path(p) for p in checkpoint_paths]
    else:
        if checkpoint_dir is None:
            raise ValueError("Provide either explicit checkpoint paths or --checkpoint-dir.")
        paths = sorted(Path(checkpoint_dir).glob(checkpoint_pattern), key=natural_key)

    if not paths:
        raise FileNotFoundError("No checkpoint files found.")
    return paths


def resolve_epoch_label(path: Path) -> str:
    match = re.search(r"epoch_(\d+)", path.stem)
    return match.group(1) if match else path.stem


def resolve_supervised_checkpoints(
    supervised_path: Path | None,
    checkpoint_pattern: str,
    epochs_filter: set[str] | None,
) -> list[Path]:
    if supervised_path is None:
        return []

    if str(supervised_path).strip() in {"", "."}:
        return []

    if supervised_path.is_file():
        checkpoints = [supervised_path]
    elif supervised_path.is_dir():
        checkpoints = sorted(supervised_path.glob(checkpoint_pattern), key=natural_key)
        if not checkpoints:
            checkpoints = sorted(supervised_path.glob("*epoch_*.pt"), key=natural_key)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {supervised_path}")
    else:
        raise FileNotFoundError(f"Supervised path does not exist: {supervised_path}")

    if epochs_filter is not None:
        filtered = [p for p in checkpoints if resolve_epoch_label(p) in epochs_filter]
        if filtered:
            checkpoints = filtered

    return checkpoints


def get_module_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if layer_name in modules:
        return modules[layer_name]
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    preview = ", ".join(sorted([name for name in modules if name])[:25])
    raise ValueError(f"Layer '{layer_name}' not found. First modules: {preview}")


def extract_layer_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    capture_output = layer_name in {"output", "model_output"}
    hook = None
    activations: dict[str, torch.Tensor] = {}

    if not capture_output:
        module = get_module_by_name(model, layer_name)

        def _hook_fn(_module, _input, output):
            activations["feat"] = output

        hook = module.register_forward_hook(_hook_fn)

    all_embeddings: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []

    try:
        with torch.no_grad():
            for images, _object_idx, sample_idx in loader:
                images = images.to(device, non_blocking=True)
                out = model(images)
                features = out if capture_output else activations.get("feat")
                if features is None:
                    raise RuntimeError(f"No activation captured for layer '{layer_name}'")

                if isinstance(features, (tuple, list)):
                    features = features[0]
                if features.ndim > 2:
                    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
                features = F.normalize(features, dim=1)

                all_embeddings.append(features.detach().cpu())
                all_indices.append(torch.as_tensor(sample_idx, dtype=torch.long))
    finally:
        if hook is not None:
            hook.remove()

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    sample_indices = torch.cat(all_indices, dim=0).numpy()
    return embeddings, sample_indices


def create_untrained_model(reference_ckpt_path: str | Path, device: torch.device) -> torch.nn.Module:
    model, _ = load_mv_model(str(reference_ckpt_path), device=str(device))
    for param in model.parameters():
        param.data.zero_()
    return model


def build_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def select_subset_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    indices = np.arange(n)
    if max_samples <= 0 or max_samples >= n:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def select_balanced_subset_indices(
    object_indices: np.ndarray,
    max_samples_per_object: int,
    seed: int,
) -> np.ndarray:
    if max_samples_per_object <= 0:
        return np.arange(len(object_indices))

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for obj_idx in np.unique(object_indices):
        candidate = np.flatnonzero(object_indices == obj_idx)
        if len(candidate) > max_samples_per_object:
            candidate = np.sort(rng.choice(candidate, size=max_samples_per_object, replace=False))
        selected.append(candidate)

    if not selected:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(selected))


def build_object_color_map(object_labels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % cmap.N) for i, label in enumerate(object_labels)}


def _scatter_by_object(
    ax: plt.Axes,
    coords: np.ndarray,
    sample_object_indices: np.ndarray,
    object_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    *,
    alpha: float,
    size: float,
    show_legend: bool,
) -> None:
    for obj_idx, label in enumerate(object_labels):
        mask = sample_object_indices == obj_idx
        if not np.any(mask):
            continue
        color = color_map.get(label, plt.get_cmap("tab20")(obj_idx % 20))
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            color=color,
            edgecolors="none",
            label=label if show_legend else None,
        )

    if show_legend:
        ax.legend(loc="best", fontsize=8, title="Object")


def fit_common_pca(
    embeddings_by_epoch: dict[str, np.ndarray],
    seed: int,
    pre_reduce_dim: int | None = None,
) -> dict[str, np.ndarray]:
    epoch_labels = list(embeddings_by_epoch.keys())
    stacked = np.concatenate([embeddings_by_epoch[e] for e in epoch_labels], axis=0)

    if pre_reduce_dim is not None and pre_reduce_dim > 0 and stacked.shape[1] > pre_reduce_dim:
        pca_pre = PCA(n_components=pre_reduce_dim, random_state=seed)
        stacked = pca_pre.fit_transform(stacked)
        print(
            f"[PCA] Reduced embeddings to {stacked.shape[1]} dims before common PCA. "
            f"Explained variance: {pca_pre.explained_variance_ratio_.sum():.4f}"
        )

    if np.var(stacked, axis=0).sum() <= 1e-12:
        print("[PCA] Degenerate embeddings detected; using zero coordinates.")
        zero_coords = np.zeros((stacked.shape[0], 2), dtype=np.float32)
        split_coords: dict[str, np.ndarray] = {}
        offset = 0
        for label in epoch_labels:
            n = embeddings_by_epoch[label].shape[0]
            split_coords[label] = zero_coords[offset : offset + n]
            offset += n
        return split_coords

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(stacked)
    print(f"[PCA] Common PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"[PCA] Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    split_coords: dict[str, np.ndarray] = {}
    offset = 0
    for label in epoch_labels:
        n = embeddings_by_epoch[label].shape[0]
        split_coords[label] = coords[offset : offset + n]
        offset += n

    return split_coords


def fit_common_pacmap(
    embeddings_by_epoch: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    epoch_labels = list(embeddings_by_epoch.keys())
    stacked = np.concatenate([embeddings_by_epoch[e] for e in epoch_labels], axis=0)

    if not PACMAP_AVAILABLE:
        raise ImportError("PACMAP is not installed. Install it with: pip install pacmap")

    if np.var(stacked, axis=0).sum() <= 1e-12:
        print("[PACMAP] Degenerate embeddings detected; using zero coordinates.")
        zero_coords = np.zeros((stacked.shape[0], 2), dtype=np.float32)
        split_coords: dict[str, np.ndarray] = {}
        offset = 0
        for label in epoch_labels:
            n = embeddings_by_epoch[label].shape[0]
            split_coords[label] = zero_coords[offset : offset + n]
            offset += n
        return split_coords

    stacked = stacked / (np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-8)
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed,
        verbose=False,
    )
    coords = reducer.fit_transform(stacked)

    split_coords: dict[str, np.ndarray] = {}
    offset = 0
    for label in epoch_labels:
        n = embeddings_by_epoch[label].shape[0]
        split_coords[label] = coords[offset : offset + n]
        offset += n

    return split_coords


def fit_common_projection(
    embeddings_by_epoch: dict[str, np.ndarray],
    use_pacmap: bool,
    seed: int,
    pre_reduce_dim: int | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    if use_pacmap:
        return fit_common_pacmap(embeddings_by_epoch=embeddings_by_epoch, seed=seed), "PACMAP"
    return (
        fit_common_pca(embeddings_by_epoch=embeddings_by_epoch, seed=seed, pre_reduce_dim=pre_reduce_dim),
        "PCA",
    )


def plot_epoch_grid(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_object_indices: np.ndarray,
    object_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
    *,
    x_tick_fontsize: int,
    y_tick_fontsize: int,
    show_legend: bool = False,
) -> None:
    n = len(epoch_labels)
    cols = 3 if n >= 3 else n
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 4.8 * rows), squeeze=False)

    all_coords = np.concatenate([coords_by_epoch[e] for e in epoch_labels], axis=0)
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    for ax in axes.ravel():
        ax.axis("off")

    for i, epoch in enumerate(epoch_labels):
        ax = axes[i // cols][i % cols]
        ax.axis("on")
        coords = coords_by_epoch[epoch]
        _scatter_by_object(
            ax,
            coords,
            sample_object_indices,
            object_labels,
            color_map,
            alpha=0.85,
            size=9,
            show_legend=False,
        )
        ax.set_title(f"Epoch {epoch}", fontsize=12)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        ax.tick_params(axis="x", labelsize=x_tick_fontsize)
        ax.tick_params(axis="y", labelsize=y_tick_fontsize)
        ax.grid(True, alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else (0, 0, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overlay(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_object_indices: np.ndarray,
    object_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
    *,
    x_tick_fontsize: int,
    y_tick_fontsize: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    all_coords = np.concatenate([coords_by_epoch[e] for e in epoch_labels], axis=0)
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    for epoch in epoch_labels:
        coords = coords_by_epoch[epoch]
        _scatter_by_object(
            ax,
            coords,
            sample_object_indices,
            object_labels,
            color_map,
            alpha=0.20,
            size=10,
            show_legend=False,
        )

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel("PC-1", fontsize=16)
    ax.set_ylabel("PC-2", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="x", labelsize=x_tick_fontsize)
    ax.tick_params(axis="y", labelsize=y_tick_fontsize)
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[label], label=label)
        for label in object_labels
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, ncol=2, title="Object")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_trajectories(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_object_indices: np.ndarray,
    object_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
    trajectory_indices: np.ndarray,
    *,
    x_tick_fontsize: int,
    y_tick_fontsize: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    all_coords = np.concatenate([coords_by_epoch[e] for e in epoch_labels], axis=0)
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    for sample_idx in trajectory_indices:
        pts = np.stack([coords_by_epoch[e][sample_idx] for e in epoch_labels], axis=0)
        obj_idx = sample_object_indices[sample_idx]
        obj_label = object_labels[obj_idx]
        color = color_map.get(obj_label, plt.get_cmap("tab20")(obj_idx % 20))
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.16, linewidth=0.9)
        ax.scatter(pts[0, 0], pts[0, 1], color=color, s=16, alpha=0.48)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=16, alpha=0.48)

    for epoch in epoch_labels:
        coords = coords_by_epoch[epoch]
        _scatter_by_object(
            ax,
            coords,
            sample_object_indices,
            object_labels,
            color_map,
            alpha=0.10,
            size=11,
            show_legend=False,
        )

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel("PC-1", fontsize=16)
    ax.set_ylabel("PC-2", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="x", labelsize=x_tick_fontsize)
    ax.tick_params(axis="y", labelsize=y_tick_fontsize)
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=color_map[label], marker="o", linestyle="", label=label)
        for label in object_labels
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, ncol=2, title="Object")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_coordinates_csv(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_indices: np.ndarray,
    object_names: Sequence[str],
    object_indices: np.ndarray,
    instance_names: Sequence[str],
    relative_paths: Sequence[str],
    filenames: Sequence[str],
    output_path: Path,
) -> None:
    rows = []
    for epoch in epoch_labels:
        coords = coords_by_epoch[epoch]
        for idx, (x, y) in enumerate(coords):
            rows.append(
                {
                    "sample_idx": int(sample_indices[idx]),
                    "subset_idx": idx,
                    "object_idx": int(object_indices[idx]),
                    "object_name": object_names[idx],
                    "instance_name": instance_names[idx],
                    "filename": filenames[idx],
                    "relative_path": relative_paths[idx],
                    "epoch": epoch,
                    "pc1": float(x),
                    "pc2": float(y),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_legend_png(
    object_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str = "Object",
    legend_fontsize: int = DEFAULT_LEGEND_FONTSIZE,
) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[label], label=label)
        for label in object_labels
    ]

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    fig.legend(handles=handles, loc="center", ncol=2, title=title, fontsize=legend_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize multi-object representations across multiple training epochs using a shared PCA/PACMAP space."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--parameter-dir",
        type=str,
        default=DEFAULT_PARAMETER_DIR,
        help="Single parameter folder to process. If omitted, all parameter folders are processed.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Optional subset of object folder names. If omitted, all objects with that parameter are used.",
    )
    parser.add_argument("--filename-col", type=str, default="image")
    parser.add_argument("--instance-glob", type=str, default="*")
    parser.add_argument("--image-subdir", type=str, default="images")
    parser.add_argument("--csv-name", type=str, default="parameters.csv")
    parser.add_argument(
        "--model-types",
        nargs="*",
        choices=["action", "ssl", "supervised"],
        default=DEFAULT_MODEL_TYPES,
        help="Model types to analyze.",
    )
    parser.add_argument("--layer", type=str, default=DEFAULT_LAYER, help="Layer to extract features from.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-arch", type=str, default=DEFAULT_MODEL_ARCH)
    parser.add_argument("--checkpoint-pattern", type=str, default="epoch_*.pt")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional explicit checkpoint path for non-supervised models.")
    parser.add_argument("--epochs", type=int, nargs="*", default=DEFAULT_EPOCHS, help="Epoch filter, e.g. 9 19 29 39 49.")
    parser.add_argument("--supervised-model", type=Path, default=DEFAULT_SUPERVISED_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on total samples used.")
    parser.add_argument(
        "--max-samples-per-object",
        type=int,
        default=0,
        help="Optional cap per object class (samples are drawn evenly across objects).",
    )
    parser.add_argument("--pca-components", type=int, default=50, help="Optional pre-reduction before the final 2D PCA.")
    parser.add_argument(
        "--use-pacmap",
        action="store_true",
        default=DEFAULT_USE_PACMAP,
        help="Use PACMAP instead of PCA for the shared embedding space (default: False).",
    )
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=150,
        help="How many sample trajectories to draw in the trajectory plot.",
    )
    parser.add_argument(
        "--x-tick-fontsize",
        type=int,
        default=DEFAULT_X_TICK_FONTSIZE,
        help="X-axis tick label font size.",
    )
    parser.add_argument(
        "--y-tick-fontsize",
        type=int,
        default=DEFAULT_Y_TICK_FONTSIZE,
        help="Y-axis tick label font size.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=int,
        default=DEFAULT_LEGEND_FONTSIZE,
        help="Legend font size for the standalone legend PNG.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pca_multi_object_epoch_outputs"))
    parser.add_argument("--include-untrained", action="store_true", default=True)
    parser.add_argument("--no-untrained", action="store_true")
    return parser.parse_args()


def process_parameter(
    dataset_root: Path,
    parameter_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print(f"\n{'=' * 90}")
    print(f"Processing parameter: {parameter_name}")
    print(f"{'=' * 90}")

    dataset = MultiObjectParameterDataset(
        dataset_root=dataset_root,
        parameter_dir=parameter_name,
        objects=args.objects,
        filename_col=args.filename_col,
        csv_name=args.csv_name,
        image_subdir=args.image_subdir,
        instance_glob=args.instance_glob,
    )

    if args.max_samples_per_object > 0:
        subset_indices = select_balanced_subset_indices(dataset.object_indices, args.max_samples_per_object, args.seed)
    else:
        subset_indices = select_subset_indices(len(dataset), args.max_samples, args.seed)

    subset = Subset(dataset, subset_indices.tolist())
    loader = build_loader(subset, batch_size=args.batch_size, num_workers=args.num_workers)

    sample_object_indices = dataset.object_indices[subset_indices]
    sample_object_names = [dataset.object_names[i] for i in subset_indices]
    sample_instance_names = [dataset.instance_names[i] for i in subset_indices]
    sample_filenames = [dataset.filenames[i] for i in subset_indices]
    sample_relative_paths = [dataset.relative_paths[i] for i in subset_indices]
    object_labels = dataset.object_labels
    color_map = build_object_color_map(object_labels)

    print(f"[Info] Dataset size: {len(dataset)}; using {len(subset_indices)} samples.")
    print(f"[Info] Objects: {object_labels}")

    epochs_filter = {str(e) for e in args.epochs} if args.epochs else None

    for model_type in args.model_types:
        embeddings_by_epoch: dict[str, np.ndarray] = {}
        used_checkpoints: list[Path] = []

        if model_type == "supervised":
            try:
                checkpoint_paths = resolve_supervised_checkpoints(args.supervised_model, args.checkpoint_pattern, epochs_filter)
            except Exception as exc:
                print(f"[Skip] Could not resolve supervised checkpoints: {exc}")
                continue
        elif args.checkpoint_path is not None:
            checkpoint_paths = [args.checkpoint_path]
        else:
            checkpoint_dir = args.model_root / args.model_arch / model_type
            try:
                checkpoint_paths = resolve_checkpoint_paths(None, checkpoint_dir, args.checkpoint_pattern)
            except Exception as exc:
                print(f"[Skip] Could not resolve checkpoints for {model_type}: {exc}")
                continue
            if epochs_filter is not None:
                checkpoint_paths = [p for p in checkpoint_paths if resolve_epoch_label(p) in epochs_filter]

        if not checkpoint_paths:
            print(f"[Skip] No checkpoints found for model type '{model_type}'.")
            continue

        print(f"\n[Group] {model_type}")
        print(f"[Info] Checkpoints: {[p.name for p in checkpoint_paths]}")

        if args.include_untrained:
            print("[Load] Creating untrained model (weights zeroed)")
            try:
                reference_ckpt = checkpoint_paths[0]
                untrained_model = create_untrained_model(reference_ckpt, device)
                embeddings, _ = extract_layer_embeddings(untrained_model, loader, device, args.layer)
                if embeddings.shape[0] != len(subset_indices):
                    raise RuntimeError(
                        f"Embedding count mismatch for untrained model: got {embeddings.shape[0]}, expected {len(subset_indices)}"
                    )
                embeddings_by_epoch["untrained"] = embeddings
                print(f"[Done] untrained: {embeddings.shape}")
            except Exception as exc:
                print(f"[Skip] Could not process untrained model: {exc}")

        for ckpt_path in checkpoint_paths:
            epoch_label = resolve_epoch_label(ckpt_path)
            print(f"[Load] {ckpt_path.name}")
            try:
                model, _ = load_mv_model(str(ckpt_path), device=str(device))
                embeddings, _ = extract_layer_embeddings(model, loader, device, args.layer)
            except Exception as exc:
                print(f"[Skip] Could not process {ckpt_path.name}: {exc}")
                continue

            if embeddings.shape[0] != len(subset_indices):
                raise RuntimeError(
                    f"Embedding count mismatch for {ckpt_path.name}: got {embeddings.shape[0]}, expected {len(subset_indices)}"
                )

            embeddings_by_epoch[epoch_label] = embeddings
            used_checkpoints.append(ckpt_path)
            print(f"[Done] epoch {epoch_label}: {embeddings.shape}")

        if len(embeddings_by_epoch) < 2:
            print(f"[Skip] Need at least two successful embeddings for '{model_type}'.")
            continue

        epoch_labels: list[str] = []
        if "untrained" in embeddings_by_epoch:
            epoch_labels.append("untrained")
        epoch_labels.extend([resolve_epoch_label(p) for p in used_checkpoints])

        coords_by_epoch, method_name = fit_common_projection(
            embeddings_by_epoch=embeddings_by_epoch,
            use_pacmap=args.use_pacmap,
            seed=args.seed,
            pre_reduce_dim=args.pca_components if args.pca_components > 0 else None,
        )

        safe_param = normalize_param_dir_name(parameter_name)
        safe_layer = args.layer.replace(".", "_")
        run_tag = "_".join(epoch_labels)

        out_dir = args.output_dir / dataset_root.name / safe_param / model_type
        out_dir.mkdir(parents=True, exist_ok=True)

        method_tag = method_name.lower()
        title = f"Multi-object {method_name} over epochs ({model_type}, {safe_param})"
        grid_path = out_dir / f"multi_object_{method_tag}_{model_type}_{safe_layer}_{safe_param}_{run_tag}_grid.png"
        overlay_path = out_dir / f"multi_object_{method_tag}_{model_type}_{safe_layer}_{safe_param}_{run_tag}_overlay.png"
        traj_path = out_dir / f"multi_object_{method_tag}_{model_type}_{safe_layer}_{safe_param}_{run_tag}_trajectories.png"
        csv_path = out_dir / f"multi_object_{method_tag}_{model_type}_{safe_layer}_{safe_param}_{run_tag}_coordinates.csv"
        legend_path = out_dir / f"multi_object_{method_tag}_{model_type}_{safe_layer}_{safe_param}_{run_tag}_legend.png"

        plot_epoch_grid(
            coords_by_epoch,
            epoch_labels,
            sample_object_indices,
            object_labels,
            color_map,
            grid_path,
            title,
            x_tick_fontsize=args.x_tick_fontsize,
            y_tick_fontsize=args.y_tick_fontsize,
        )
        plot_overlay(
            coords_by_epoch,
            epoch_labels,
            sample_object_indices,
            object_labels,
            color_map,
            overlay_path,
            title + " — overlay",
            x_tick_fontsize=args.x_tick_fontsize,
            y_tick_fontsize=args.y_tick_fontsize,
        )

        traj_n = min(args.trajectory_samples, len(subset_indices))
        traj_indices = np.sort(np.random.default_rng(args.seed).choice(len(subset_indices), size=traj_n, replace=False))
        plot_trajectories(
            coords_by_epoch,
            epoch_labels,
            sample_object_indices,
            object_labels,
            color_map,
            traj_path,
            title + " — trajectories",
            traj_indices,
            x_tick_fontsize=args.x_tick_fontsize,
            y_tick_fontsize=args.y_tick_fontsize,
        )

        save_coordinates_csv(
            coords_by_epoch,
            epoch_labels,
            subset_indices,
            sample_object_names,
            sample_object_indices,
            sample_instance_names,
            sample_relative_paths,
            sample_filenames,
            csv_path,
        )

        save_legend_png(
            object_labels,
            color_map,
            legend_path,
            legend_fontsize=args.legend_fontsize,
        )

        print(f"[Info] Projection method: {method_name}")
        print("[Saved]")
        print(f"  {grid_path}")
        print(f"  {overlay_path}")
        print(f"  {traj_path}")
        print(f"  {legend_path}")
        print(f"  {csv_path}")


def main() -> None:
    args = parse_args()
    if args.no_untrained:
        args.include_untrained = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Using device: {device}")
    print(f"[Info] Dataset root: {args.dataset_root}")
    print(f"[Info] Model root: {args.model_root / args.model_arch}")
    print(f"[Info] Model types: {args.model_types}")
    print(f"[Info] Layer: {args.layer}")
    print(f"[Info] Include untrained: {args.include_untrained}")
    print(f"[Info] Dimensionality reduction: {'PACMAP' if args.use_pacmap else 'PCA'}")

    parameter_dirs = [args.parameter_dir] if args.parameter_dir else collect_parameter_dirs(args.dataset_root, args.objects)
    if not parameter_dirs:
        raise RuntimeError(f"No parameter folders found under {args.dataset_root}")

    print(f"[Info] Parameter folders: {parameter_dirs}")
    for parameter_name in parameter_dirs:
        try:
            process_parameter(args.dataset_root, parameter_name, args, device)
        except Exception as exc:
            print(f"[Error] Failed to process '{parameter_name}': {exc}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()

