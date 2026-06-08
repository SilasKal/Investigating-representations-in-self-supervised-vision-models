r"""PaCMAP evolution plots for a sequence of training checkpoints.

This script is intended for comparing checkpoint sequences such as
`epoch_9.pt`, `epoch_19.pt`, ..., `epoch_49.pt` and visualizing how the
last-layer representations change over training.

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
import pacmap
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from pretrained.load_mvimgnet_model import load_mv_model


DEFAULT_DATASET_ROOT = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_new_multiple")
DEFAULT_MODEL_ROOT = Path("model_files")
DEFAULT_MODEL_ARCH = "resnet18"
DEFAULT_MODEL_GROUPS = ["action", "ssl"]
DEFAULT_EPOCHS = [9, 19, 29, 39, 49]
DEFAULT_CLASS_GLOB = "*"
DEFAULT_INSTANCE_GLOB = "*"
DEFAULT_IMAGE_SUBDIR = "images"
DEFAULT_CSV_NAME = "parameters.csv"
DEFAULT_SPLIT_MODE = "last1"


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else math.inf, path.name)


class ImageOnlyDataset(Dataset):
    """Dataset that aggregates class folders and selected instance folders.

    Expected structure:
      root/
        class_a/
          class_a_001/
            parameters.csv
            images/
          class_a_002/
            parameters.csv
            images/
        ...

    The CSV must contain a filename column (default: `image`).
    """

    def __init__(
        self,
        root: str | Path,
        filename_col: str = "image",
        split_mode: str = DEFAULT_SPLIT_MODE,
        class_glob: str = DEFAULT_CLASS_GLOB,
        instance_glob: str = DEFAULT_INSTANCE_GLOB,
        image_subdir: str = DEFAULT_IMAGE_SUBDIR,
        csv_name: str = DEFAULT_CSV_NAME,
        transform=None,
        drop_missing_files: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split_mode = split_mode
        self.class_glob = class_glob
        self.instance_glob = instance_glob
        self.image_subdir = image_subdir
        self.csv_name = csv_name
        self.transform = transform

        if split_mode not in {"train4", "last1"}:
            raise ValueError("split_mode must be one of: train4, last1")

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        class_dirs = sorted([p for p in self.root.glob(self.class_glob) if p.is_dir()], key=natural_key)
        self._single_instance_mode = False
        if not class_dirs and (self.root / self.csv_name).exists() and (self.root / self.image_subdir).exists():
            class_dirs = [self.root]
            self._single_instance_mode = True

        if not class_dirs:
            raise FileNotFoundError(f"No class folders found under: {self.root}")

        records: list[dict[str, object]] = []
        class_names: list[str] = []
        for class_idx, class_dir in enumerate(class_dirs):
            if self._single_instance_mode:
                instance_selection = [(class_dir, 1)]
            else:
                candidate_instances = sorted(
                    [p for p in class_dir.glob(self.instance_glob) if p.is_dir()],
                    key=natural_key,
                )
                valid_instances = [p for p in candidate_instances if (p / self.csv_name).exists() and (p / self.image_subdir).exists()]
                if not valid_instances:
                    raise FileNotFoundError(
                        f"No valid instance folders found under class folder: {class_dir}"
                    )
                if split_mode == "train4":
                    if len(valid_instances) < 4:
                        raise ValueError(
                            f"Class '{class_dir.name}' has only {len(valid_instances)} instances, but split_mode=train4 requires at least 4."
                        )
                    instance_selection = [(inst, rank + 1) for rank, inst in enumerate(valid_instances[:4])]
                else:
                    instance_selection = [(valid_instances[-1], len(valid_instances))]

            class_names.append(class_dir.name)
            for instance_dir, instance_rank_in_class in instance_selection:
                csv_path = instance_dir / self.csv_name
                img_dir = instance_dir / self.image_subdir
                if not csv_path.exists():
                    raise FileNotFoundError(f"Missing CSV: {csv_path}")
                if not img_dir.exists():
                    raise FileNotFoundError(f"Missing image folder: {img_dir}")

                df = pd.read_csv(csv_path)
                if filename_col not in df.columns:
                    raise ValueError(
                        f"CSV missing filename column '{filename_col}' in {csv_path}. Found: {list(df.columns)}"
                    )

                df = df.reset_index(drop=True)
                for local_idx, name in enumerate(df[filename_col].astype(str)):
                    filename = str(name)
                    img_path = img_dir / filename
                    if drop_missing_files and not img_path.exists():
                        continue
                    records.append(
                        {
                            "class_name": class_dir.name,
                            "class_idx": class_idx,
                            "instance_name": instance_dir.name,
                            "instance_rank_in_class": instance_rank_in_class,
                            "instance_idx": len(records),
                            "local_idx": local_idx,
                            "filename": img_path.name,
                            "relative_path": str(img_path.relative_to(self.root)),
                            "path": img_path,
                        }
                    )

        if not records:
            raise RuntimeError(f"No images found under {self.root}.")

        self.df = pd.DataFrame(records)
        self.paths = [Path(p) for p in self.df["path"].tolist()]
        self.filenames = self.df["filename"].tolist()
        self.class_names = self.df["class_name"].tolist()
        self.class_indices = self.df["class_idx"].to_numpy(dtype=np.int64)
        self.instance_names = self.df["instance_name"].tolist()
        self.instance_rank_in_class = self.df["instance_rank_in_class"].to_numpy(dtype=np.int64)
        self.relative_paths = self.df["relative_path"].tolist()
        self.filename_col = filename_col
        self.class_labels = class_names
        self.split_mode = split_mode

        if self.transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, idx


def natural_epoch_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.stem)
    return (int(match.group(1)) if match else math.inf, path.name)


def resolve_checkpoints(
    checkpoint_paths: Sequence[str | Path] | None,
    checkpoint_dir: str | Path | None,
    checkpoint_pattern: str,
) -> list[Path]:
    if checkpoint_paths:
        paths = [Path(p) for p in checkpoint_paths]
    else:
        if checkpoint_dir is None:
            raise ValueError("Provide either explicit checkpoint paths or --checkpoint-dir.")
        paths = sorted(Path(checkpoint_dir).glob(checkpoint_pattern), key=natural_epoch_key)

    if not paths:
        raise FileNotFoundError("No checkpoint files found.")

    return paths


def resolve_epoch_label(path: Path) -> str:
    match = re.search(r"epoch_(\d+)", path.stem)
    return match.group(1) if match else path.stem


def build_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def select_subset_indices(n: int, max_samples: int | None, seed: int) -> np.ndarray:
    indices = np.arange(n)
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return indices
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(indices, size=max_samples, replace=False))
    return chosen


def select_balanced_subset_indices(
    group_indices: np.ndarray,
    max_samples_per_instance: int,
    seed: int,
) -> np.ndarray:
    if max_samples_per_instance is None or max_samples_per_instance <= 0:
        return np.arange(len(group_indices))

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for group_idx in np.unique(group_indices):
        candidate = np.flatnonzero(group_indices == group_idx)
        if len(candidate) > max_samples_per_instance:
            candidate = np.sort(rng.choice(candidate, size=max_samples_per_instance, replace=False))
        selected.append(candidate)

    if not selected:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(selected))


def build_class_color_map(class_labels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % cmap.N) for i, label in enumerate(class_labels)}


def _scatter_by_instance(
    ax: plt.Axes,
    coords: np.ndarray,
    sample_group_indices: np.ndarray,
    group_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    *,
    alpha: float,
    size: float,
    show_legend: bool,
) -> None:
    for group_idx, label in enumerate(group_labels):
        mask = sample_group_indices == group_idx
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            alpha=alpha,
            color=color_map[label],
            edgecolors="none",
            label=label if show_legend else None,
        )

    if show_legend:
        ax.legend(loc="best", fontsize=8, title="Instance")


def _forward_backbone(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return last-layer representations before the classification head."""
    if all(hasattr(model, name) for name in ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]):
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        x = backbone(images)
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    x = model(images)
    if x.ndim == 4:
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return x


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    all_embeddings: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            images, indices = batch
            images = images.to(device, non_blocking=True)
            embeddings = _forward_backbone(model, images)
            embeddings = F.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings.cpu())
            all_indices.append(torch.as_tensor(indices, dtype=torch.long))

    if not all_embeddings:
        raise RuntimeError("No embeddings were extracted; check the dataset loader.")

    emb = torch.cat(all_embeddings, dim=0).numpy()
    idx = torch.cat(all_indices, dim=0).numpy()
    return emb, idx


def fit_common_pacmap(
    embeddings_by_epoch: dict[str, np.ndarray],
    pca_components: int | None,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    epoch_labels = list(embeddings_by_epoch.keys())
    stacked = np.concatenate([embeddings_by_epoch[e] for e in epoch_labels], axis=0)

    if pca_components is not None and stacked.shape[1] > pca_components:
        pca = PCA(n_components=pca_components, random_state=seed)
        stacked = pca.fit_transform(stacked)
        print(
            f"[PCA] Reduced embeddings to {stacked.shape[1]} dims before PaCMAP. "
            f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}"
        )

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed,
    )
    coords = reducer.fit_transform(stacked)

    split_coords: dict[str, np.ndarray] = {}
    offset = 0
    for label in epoch_labels:
        n = embeddings_by_epoch[label].shape[0]
        split_coords[label] = coords[offset : offset + n]
        offset += n

    return split_coords, coords


def plot_epoch_grid(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_instance_indices: np.ndarray,
    instance_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
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
        _scatter_by_instance(
            ax,
            coords,
            sample_instance_indices,
            instance_labels,
            color_map,
            alpha=0.85,
            size=9,
            show_legend=False,
        )
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_xlabel("PaCMAP-1")
        ax.set_ylabel("PaCMAP-2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overlay(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_instance_indices: np.ndarray,
    instance_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    all_coords = np.concatenate([coords_by_epoch[e] for e in epoch_labels], axis=0)
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    for epoch in epoch_labels:
        coords = coords_by_epoch[epoch]
        _scatter_by_instance(
            ax,
            coords,
            sample_instance_indices,
            instance_labels,
            color_map,
            alpha=0.20,
            size=10,
            show_legend=False,
        )

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[label], label=label)
        for label in instance_labels
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, ncol=2, title="Instance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_trajectories(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_instance_indices: np.ndarray,
    instance_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
    trajectory_indices: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    all_coords = np.concatenate([coords_by_epoch[e] for e in epoch_labels], axis=0)
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    for sample_idx in trajectory_indices:
        pts = np.stack([coords_by_epoch[e][sample_idx] for e in epoch_labels], axis=0)
        inst_idx = sample_instance_indices[sample_idx]
        inst_label = instance_labels[inst_idx]
        color = color_map[inst_label]
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.16, linewidth=0.9)
        ax.scatter(pts[0, 0], pts[0, 1], color=color, s=16, alpha=0.48)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=16, alpha=0.48)

    for epoch in epoch_labels:
        coords = coords_by_epoch[epoch]
        _scatter_by_instance(
            ax,
            coords,
            sample_instance_indices,
            instance_labels,
            color_map,
            alpha=0.10,
            size=11,
            show_legend=False,
        )

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=color_map[label], marker="o", linestyle="", label=label)
        for label in instance_labels
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, ncol=2, title="Instance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_coordinates_csv(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_indices: np.ndarray,
    class_names: Sequence[str],
    class_indices: np.ndarray,
    instance_names: Sequence[str],
    instance_rank_in_class: np.ndarray,
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
                    "class_idx": int(class_indices[idx]),
                    "class_name": class_names[idx],
                    "instance_name": instance_names[idx],
                    "instance_rank_in_class": int(instance_rank_in_class[idx]),
                    "filename": filenames[idx],
                    "relative_path": relative_paths[idx],
                    "epoch": epoch,
                    "pacmap_1": float(x),
                    "pacmap_2": float(y),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize representation evolution across training checkpoints using PaCMAP."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing parameters.csv and images/. Can be changed later.",
    )
    parser.add_argument(
        "--filename-col",
        type=str,
        default="image",
        help="Name of the CSV column that contains the image filename.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["train4", "last1"],
        default=DEFAULT_SPLIT_MODE,
        help="Which instance split to load per class: first 4 instances or only the last instance.",
    )
    parser.add_argument(
        "--class-glob",
        type=str,
        default=DEFAULT_CLASS_GLOB,
        help="Glob used to discover class folders under --dataset-root.",
    )
    parser.add_argument(
        "--instance-glob",
        type=str,
        default=DEFAULT_INSTANCE_GLOB,
        help="Glob used to discover instance folders inside each class folder.",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default=DEFAULT_IMAGE_SUBDIR,
        help="Image subfolder name inside each instance directory.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=DEFAULT_CSV_NAME,
        help="CSV filename inside each instance directory.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Root directory containing model folders like model_files/resnet18/action.",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default=DEFAULT_MODEL_ARCH,
        help="Model architecture subfolder inside --model-root (default: resnet18).",
    )
    parser.add_argument(
        "--model-groups",
        type=str,
        nargs="*",
        default=DEFAULT_MODEL_GROUPS,
        help="Model group subfolders to plot, e.g. action ssl.",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="epoch_*.pt",
        help="Glob pattern used inside each model group directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=DEFAULT_EPOCHS,
        help="Epoch filter. Only checkpoints whose filename contains one of these epochs are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pacmap_evolution_outputs"),
        help="Directory where per-group plots and CSV files will be written.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of samples used for the visualization.",
    )
    parser.add_argument(
        "--max-samples-per-instance",
        type=int,
        default=0,
        help="Optional per-class cap. If > 0, samples are drawn evenly from each class.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Optional PCA dimensionality before PaCMAP. Set to 0 to disable.",
    )
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=150,
        help="How many sample trajectories to draw in the development plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Using device: {device}")
    print(f"[Info] Dataset root: {args.dataset_root}")
    print(f"[Info] Model root: {args.model_root / args.model_arch}")
    print(f"[Info] Model groups: {args.model_groups}")

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = ImageOnlyDataset(
        args.dataset_root,
        filename_col=args.filename_col,
        split_mode=args.split_mode,
        class_glob=args.class_glob,
        instance_glob=args.instance_glob,
        image_subdir=args.image_subdir,
        csv_name=args.csv_name,
        transform=transform,
    )

    if args.max_samples_per_instance > 0:
        subset_indices = select_balanced_subset_indices(dataset.class_indices, args.max_samples_per_instance, args.seed)
    else:
        subset_indices = select_subset_indices(len(dataset), args.max_samples, args.seed)

    subset = Subset(dataset, subset_indices.tolist())
    filenames = [dataset.filenames[i] for i in subset_indices]
    relative_paths = [dataset.relative_paths[i] for i in subset_indices]
    sample_class_indices = dataset.class_indices[subset_indices]
    sample_class_names = [dataset.class_names[i] for i in subset_indices]
    sample_instance_names = [dataset.instance_names[i] for i in subset_indices]
    sample_instance_rank_in_class = dataset.instance_rank_in_class[subset_indices]
    class_labels = dataset.class_labels
    color_map = build_class_color_map(class_labels)
    loader = build_loader(subset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"[Info] Dataset size: {len(dataset)}; using {len(subset_indices)} samples for analysis.")
    epochs_filter = {str(e) for e in args.epochs} if args.epochs else None

    for group_name in args.model_groups:
        checkpoint_dir = args.model_root / args.model_arch / group_name
        checkpoint_paths = resolve_checkpoints(None, checkpoint_dir, args.checkpoint_pattern)
        if epochs_filter is not None:
            checkpoint_paths = [p for p in checkpoint_paths if resolve_epoch_label(p) in epochs_filter]

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoints matched the requested epochs/filter in {checkpoint_dir}."
            )

        print(f"\n[Group] {group_name}")
        print(f"[Info] Checkpoints: {[str(p) for p in checkpoint_paths]}")

        embeddings_by_epoch: dict[str, np.ndarray] = {}
        used_checkpoints: list[Path] = []

        for ckpt_path in checkpoint_paths:
            epoch_label = resolve_epoch_label(ckpt_path)
            print(f"[Load] {ckpt_path.name}")
            try:
                model, _ = load_mv_model(str(ckpt_path), device=str(device))
                embeddings, _sample_indices = extract_embeddings(model, loader, device)
            except Exception as exc:
                print(f"[Skip] Could not process {ckpt_path.name}: {exc}")
                continue

            if embeddings.shape[0] != len(subset_indices):
                raise RuntimeError(
                    f"Embedding count mismatch for {ckpt_path.name}: got {embeddings.shape[0]}, expected {len(subset_indices)}"
                )

            embeddings_by_epoch[epoch_label] = embeddings
            used_checkpoints.append(ckpt_path)
            print(f"[Done] epoch {epoch_label}: embeddings shape = {embeddings.shape}")

        if len(embeddings_by_epoch) < 2:
            raise RuntimeError(
                f"Need at least two successful checkpoints for group '{group_name}' to visualize training development."
            )

        epoch_labels = [resolve_epoch_label(p) for p in used_checkpoints]

        coords_by_epoch, _ = fit_common_pacmap(
            embeddings_by_epoch=embeddings_by_epoch,
            pca_components=args.pca_components if args.pca_components > 0 else None,
            seed=args.seed,
        )

        group_output_dir = args.output_dir / args.model_arch / args.split_mode / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"pacmap_evolution_{args.model_arch}_{args.split_mode}_{group_name}"
        grid_path = group_output_dir / f"{prefix}_grid.png"
        overlay_path = group_output_dir / f"{prefix}_overlay.png"
        traj_path = group_output_dir / f"{prefix}_trajectories.png"
        csv_path = group_output_dir / f"{prefix}_coordinates.csv"

        title = f"PaCMAP evolution of the last-layer representations ({args.model_arch} / {group_name})"
        plot_epoch_grid(coords_by_epoch, epoch_labels, sample_class_indices, class_labels, color_map, grid_path, title)
        plot_overlay(
            coords_by_epoch,
            epoch_labels,
            sample_class_indices,
            class_labels,
            color_map,
            overlay_path,
            title + " — overlay",
        )

        traj_n = min(args.trajectory_samples, len(subset_indices))
        traj_indices = np.sort(np.random.default_rng(args.seed).choice(len(subset_indices), size=traj_n, replace=False))
        plot_trajectories(
            coords_by_epoch,
            epoch_labels,
            sample_class_indices,
            class_labels,
            color_map,
            traj_path,
            title + " — sample trajectories",
            traj_indices,
        )

        save_coordinates_csv(
            coords_by_epoch,
            epoch_labels,
            subset_indices,
            sample_class_names,
            sample_class_indices,
            sample_instance_names,
            sample_instance_rank_in_class,
            relative_paths,
            filenames,
            csv_path,
        )

        print("[Saved]")
        print(f"  {grid_path}")
        print(f"  {overlay_path}")
        print(f"  {traj_path}")
        print(f"  {csv_path}")


if __name__ == "__main__":
    main()


