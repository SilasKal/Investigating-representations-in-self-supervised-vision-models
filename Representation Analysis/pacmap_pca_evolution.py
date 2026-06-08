r"""PCA/PACMAP projection plots for a sequence of training checkpoints.

This script compares checkpoint sequences such as
`epoch_9.pt`, `epoch_19.pt`, ..., `epoch_49.pt` and visualizing how the
last-layer representations change over training using dimensionality reduction.
You can choose between PCA (Principal Component Analysis) or PACMAP by using the --use-pacmap flag.
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
DEFAULT_MODEL_ROOT = Path("model_files")
DEFAULT_MODEL_ARCH = "resnet18"
DEFAULT_EPOCHS = [9, 19, 29, 39, 49]
DEFAULT_INSTANCE_GLOB = "*"
DEFAULT_IMAGE_SUBDIR = "images"
DEFAULT_CSV_NAME = "parameters.csv"
DEFAULT_USE_PACMAP = True  # Use PCA by default, set to True to use PACMAP instead#
# SUPERVISED_MODEL_PATH = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\supervised")
SUPERVISED_MODEL_PATH = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\supervised_MAPS_resnet18_seed0_1.pt")


# ============================================================
# GLOBAL PLOT STYLE SETTINGS
# Change all plot sizes here
# ============================================================

PLOT_DPI_GRID = 200
PLOT_DPI_SINGLE = 220

AXIS_LABEL_FONTSIZE = 100
TICK_LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
PANEL_TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 8

GRID_SCATTER_SIZE = 9
OVERLAY_SCATTER_SIZE = 10
TRAJECTORY_SCATTER_SIZE = 11
TRAJECTORY_POINT_SIZE = 16
TRAJECTORY_LINEWIDTH = 0.9

GRID_ALPHA = 0.85
OVERLAY_ALPHA = 0.20
TRAJECTORY_ALPHA = 0.10
TRAJECTORY_LINE_ALPHA = 0.16
TRAJECTORY_POINT_ALPHA = 0.48

GRID_SINGLE_FIGSIZE = (8.8, 7.6)
GRID_PANEL_FIGSIZE = (5.3, 4.8)
SINGLE_PLOT_FIGSIZE = (10, 8)

def natural_instance_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else math.inf, path.name)


class ImageOnlyDataset(Dataset):
    """Dataset that aggregates all instance folders under a root directory.

    Supports two structures:
    
    Old structure (flat):
      root/
        instance_001/
          parameters.csv
          images/
        instance_002/
          parameters.csv
          images/
        ...
    
    New structure (hierarchical):
      root/
        object_name/
          parameter_name/
            instance_1/
              parameters.csv
              images/
            instance_2/
              parameters.csv
              images/
            ...

    The CSV must contain a filename column (default: `image`). All instances are
    concatenated into a single ordered dataset so PCA can be computed across
    instances in a shared coordinate system.
    """

    def __init__(
        self,
        root: str | Path,
        filename_col: str = "image",
        instance_glob: str = DEFAULT_INSTANCE_GLOB,
        image_subdir: str = DEFAULT_IMAGE_SUBDIR,
        csv_name: str = DEFAULT_CSV_NAME,
        transform=None,
        drop_missing_files: bool = True,
        object_filter: str | None = None,
        parameter_filter: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.instance_glob = instance_glob
        self.image_subdir = image_subdir
        self.csv_name = csv_name
        self.transform = transform
        self.object_filter = object_filter
        self.parameter_filter = parameter_filter

        if (self.root / self.csv_name).exists() and (self.root / self.image_subdir).exists():
            instance_dirs = [self.root]
        else:
            # Try flat structure first
            instance_dirs = sorted(
                [p for p in self.root.glob(self.instance_glob) if p.is_dir() and (p / self.csv_name).exists()],
                key=natural_instance_key,
            )
            
            # If no instances found, try hierarchical structure (object_name/parameter_name/instance_*)
            if not instance_dirs:
                instance_dirs = self._collect_hierarchical_instances()
        
        if not instance_dirs:
            raise FileNotFoundError(f"No instance folders found under: {self.root}")

        records: list[dict[str, object]] = []
        for instance_idx, instance_dir in enumerate(instance_dirs):
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
                        "instance_name": instance_dir.name,
                        "instance_idx": instance_idx,
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
        self.instance_names = self.df["instance_name"].tolist()
        self.instance_indices = self.df["instance_idx"].to_numpy(dtype=np.int64)
        self.relative_paths = self.df["relative_path"].tolist()
        self.filename_col = filename_col
        self.instance_dirs = instance_dirs
        self.instance_labels = [p.name for p in instance_dirs]

        if self.transform is None:
            self.transform = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    def _collect_hierarchical_instances(self) -> list[Path]:
        """Collect instances from hierarchical structure: object_name/parameter_name/instance_*/
        
        If object_filter and/or parameter_filter are set, only collects instances matching those criteria.
        Returns sorted list of instance directories.
        """
        instance_dirs = []
        
        # Iterate through object_name directories
        for object_dir in sorted(self.root.iterdir()):
            if not object_dir.is_dir():
                continue
            
            # Skip this object if we're filtering and it doesn't match
            if self.object_filter is not None and object_dir.name != self.object_filter:
                continue
            
            # Iterate through parameter_name directories
            for param_dir in sorted(object_dir.iterdir()):
                if not param_dir.is_dir():
                    continue
                
                # Skip this parameter if we're filtering and it doesn't match
                if self.parameter_filter is not None and param_dir.name != self.parameter_filter:
                    continue
                
                # Collect instance directories
                instances = sorted(
                    [p for p in param_dir.glob(self.instance_glob) if p.is_dir() and (p / self.csv_name).exists()],
                    key=natural_instance_key,
                )
                instance_dirs.extend(instances)
        
        return instance_dirs


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, idx


def natural_epoch_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)", path.stem)
    return (int(match.group(1)) if match else math.inf, path.name)


def collect_object_names(dataset_root: Path) -> list[str]:
    """Collect all unique object names from hierarchical dataset structure.
    
    Returns sorted list of object names.
    """
    objects = []
    
    # Iterate through object_name directories
    for object_dir in sorted(dataset_root.iterdir()):
        if object_dir.is_dir():
            objects.append(object_dir.name)
    
    return sorted(objects)


def collect_parameter_names_for_object(dataset_root: Path, object_name: str) -> list[str]:
    """Collect all parameter names for a specific object.
    
    Returns sorted list of parameter names.
    """
    parameters = []
    object_dir = dataset_root / object_name
    
    if not object_dir.exists():
        return parameters
    
    # Iterate through parameter_name directories
    for param_dir in sorted(object_dir.iterdir()):
        if param_dir.is_dir():
            parameters.append(param_dir.name)
    
    return sorted(parameters)


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


def resolve_supervised_checkpoints(
    supervised_path: Path | None,
    checkpoint_pattern: str,
    epochs_filter: set[str] | None,
) -> list[Path]:
    """Resolve supervised checkpoints from either a single file or a directory."""
    if supervised_path is None:
        return []

    # Historically users disabled supervised loading with an empty string, which
    # becomes "." after argparse Path conversion.
    if str(supervised_path).strip() in {"", "."}:
        return []

    if supervised_path.is_file():
        checkpoints = [supervised_path]
    elif supervised_path.is_dir():
        # First try the user/default pattern, then fall back to a broad pattern
        # to support names like "supervised_..._epoch_0010.pt".
        checkpoints = sorted(supervised_path.glob(checkpoint_pattern), key=natural_epoch_key)
        if not checkpoints:
            checkpoints = sorted(supervised_path.glob("*epoch_*.pt"), key=natural_epoch_key)
        if not checkpoints:
            raise FileNotFoundError("No checkpoint files found.")
    else:
        raise FileNotFoundError(f"Supervised path does not exist: {supervised_path}")

    all_checkpoints = checkpoints
    if epochs_filter is not None:
        checkpoints = [p for p in checkpoints if resolve_epoch_label(p) in epochs_filter]
        # If filtering removes all supervised checkpoints, fall back to "all" so
        # supervised still works with directory layouts that use different epochs.
        if not checkpoints:
            checkpoints = all_checkpoints

    return checkpoints


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
    instance_indices: np.ndarray,
    max_samples_per_instance: int,
    seed: int,
) -> np.ndarray:
    if max_samples_per_instance is None or max_samples_per_instance <= 0:
        return np.arange(len(instance_indices))

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for instance_idx in np.unique(instance_indices):
        candidate = np.flatnonzero(instance_indices == instance_idx)
        if len(candidate) > max_samples_per_instance:
            candidate = np.sort(rng.choice(candidate, size=max_samples_per_instance, replace=False))
        selected.append(candidate)

    if not selected:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(selected))


def build_instance_color_map(instance_labels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % cmap.N) for i, label in enumerate(instance_labels)}


def _scatter_by_instance(
    ax: plt.Axes,
    coords: np.ndarray,
    sample_instance_indices: np.ndarray,
    instance_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    *,
    alpha: float,
    size: float,
    show_legend: bool,
) -> None:
    for instance_idx, label in enumerate(instance_labels):
        mask = sample_instance_indices == instance_idx
        if not np.any(mask):
            continue
        cmap = plt.get_cmap("tab10")
        color = color_map.get(label, cmap(instance_idx % cmap.N))
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


def create_untrained_model(
    reference_ckpt_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Create an untrained model by loading a checkpoint and zeroing all weights."""
    model, _ = load_mv_model(str(reference_ckpt_path), device=str(device))
    
    # Zero out all parameters
    for param in model.parameters():
        param.data.zero_()
    
    return model


def fit_pca(
    embeddings_by_epoch: dict[str, np.ndarray],
    pca_components: int | None,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit PCA independently for each epoch."""
    epoch_labels = list(embeddings_by_epoch.keys())
    
    # Fit PCA separately for each epoch
    split_coords: dict[str, np.ndarray] = {}
    all_coords_list = []
    
    for label in epoch_labels:
        embeddings = embeddings_by_epoch[label]
        print(f"[PCA] Fitting PCA for epoch '{label}' on {embeddings.shape[0]} samples with {embeddings.shape[1]} dims...")

        # Degenerate case: all embeddings are effectively identical (e.g.,
        # zero-initialized untrained model). PCA has undefined explained variance.
        if np.var(embeddings, axis=0).sum() <= 1e-12:
            coords = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
            print(f"[PCA] Epoch '{label}' is degenerate (near-zero variance); using zero coordinates.")
            print("[PCA] Explained variance ratio: [0. 0.]")
            print("[PCA] Total explained variance: 0.0000")
            split_coords[label] = coords
            all_coords_list.append(coords)
            continue
        
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(embeddings)
        
        print(f"[PCA] Epoch '{label}' completed. Output shape: {coords.shape}")
        print(f"[PCA] Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"[PCA] Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        split_coords[label] = coords
        all_coords_list.append(coords)
    
    # Combine all coordinates for reference (each epoch in its own space)
    all_coords = np.vstack(all_coords_list) if all_coords_list else np.array([])

    return split_coords, all_coords


def fit_pacmap_method(
    embeddings_by_epoch: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit PACMAP independently for each epoch."""
    if not PACMAP_AVAILABLE:
        raise ImportError(
            "PACMAP is not installed. Install it with: pip install pacmap"
        )
    
    epoch_labels = list(embeddings_by_epoch.keys())
    
    # Fit PACMAP separately for each epoch
    split_coords: dict[str, np.ndarray] = {}
    all_coords_list = []
    
    for label in epoch_labels:
        embeddings = embeddings_by_epoch[label]
        print(f"[PACMAP] Fitting PACMAP for epoch '{label}' on {embeddings.shape[0]} samples with {embeddings.shape[1]} dims...")

        # Degenerate case: all embeddings are effectively identical
        if np.var(embeddings, axis=0).sum() <= 1e-12:
            coords = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
            print(f"[PACMAP] Epoch '{label}' is degenerate (near-zero variance); using zero coordinates.")
            split_coords[label] = coords
            all_coords_list.append(coords)
            continue
        
        try:
            # PACMAP requires normalized embeddings for best results
            embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            
            # Initialize PACMAP with PCA for faster convergence
            reducer = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=10,
                MN_ratio=0.5,
                FP_ratio=2.0,
                random_state=seed,
                verbose=False,
            )
            coords = reducer.fit_transform(embeddings_normalized)
            
            print(f"[PACMAP] Epoch '{label}' completed. Output shape: {coords.shape}")
            
            split_coords[label] = coords
            all_coords_list.append(coords)
        except Exception as e:
            print(f"[PACMAP] Error fitting PACMAP for epoch '{label}': {e}")
            # Fallback to zero coordinates
            coords = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
            split_coords[label] = coords
            all_coords_list.append(coords)
    
    # Combine all coordinates for reference (each epoch in its own space)
    all_coords = np.vstack(all_coords_list) if all_coords_list else np.array([])

    return split_coords, all_coords


def fit_common_embedding_projection(
    embeddings_by_epoch: dict[str, np.ndarray],
    use_pacmap: bool,
    pca_components: int | None,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    """Fit either PCA or PACMAP based on the use_pacmap flag.
    
    Returns:
        - coords_by_epoch: 2D coordinates for each epoch
        - all_coords: concatenated coordinates
        - method_name: the method used ("PCA" or "PACMAP")
    """
    if use_pacmap:
        coords_by_epoch, all_coords = fit_pacmap_method(
            embeddings_by_epoch=embeddings_by_epoch,
            seed=seed,
        )
        return coords_by_epoch, all_coords, "PACMAP"
    else:
        coords_by_epoch, all_coords = fit_pca(
            embeddings_by_epoch=embeddings_by_epoch,
            pca_components=pca_components if pca_components > 0 else None,
            seed=seed,
        )
        return coords_by_epoch, all_coords, "PCA"


def plot_epoch_grid(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_instance_indices: np.ndarray,
    instance_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
    *,
    show_legend: bool = False,
    axis_fontsize: int = AXIS_LABEL_FONTSIZE,
    tick_fontsize: int = TICK_LABEL_FONTSIZE,
    title_fontsize: int = TITLE_FONTSIZE,
    panel_title_fontsize: int = PANEL_TITLE_FONTSIZE,
    legend_fontsize: int = LEGEND_FONTSIZE,
    scatter_size: int = GRID_SCATTER_SIZE,
    x_label: str = "Dim-1",
    y_label: str = "Dim-2",
    show_title: bool = True,
    show_panel_titles: bool = False,
) -> None:
    n = len(epoch_labels)
    cols = 3 if n >= 3 else n
    rows = math.ceil(n / cols)

    if n == 1:
        fig_size = GRID_SINGLE_FIGSIZE
    else:
        fig_size = (GRID_PANEL_FIGSIZE[0] * cols, GRID_PANEL_FIGSIZE[1] * rows)

    fig, axes = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)

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
            alpha=GRID_ALPHA,
            size=scatter_size,
            show_legend=False,
        )

        if show_panel_titles:
            ax.set_title(f"Epoch {epoch}", fontsize=panel_title_fontsize)

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_xlabel(x_label, fontsize=axis_fontsize)
        ax.set_ylabel(y_label, fontsize=axis_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(True, alpha=0.2)

    if show_title and title:
        fig.suptitle(title, fontsize=title_fontsize)

    if show_legend:
        from matplotlib.lines import Line2D

        cmap = plt.get_cmap("tab10")
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color_map.get(label, cmap(i % cmap.N)),
                label=label,
            )
            for i, label in enumerate(instance_labels)
        ]

        fig.tight_layout(rect=(0, 0, 0.92, 0.96) if show_title and title else (0, 0, 0.92, 1))

        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="center right",
                fontsize=legend_fontsize,
                title="Instance",
            )
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96) if show_title and title else (0, 0, 1, 1))

    fig.savefig(output_path, dpi=PLOT_DPI_GRID)
    plt.close(fig)


def plot_overlay(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_instance_indices: np.ndarray,
    instance_labels: Sequence[str],
    color_map: dict[str, tuple[float, float, float, float]],
    output_path: Path,
    title: str,
    *,
    show_legend: bool = True,
    axis_fontsize: int = AXIS_LABEL_FONTSIZE,
    tick_fontsize: int = TICK_LABEL_FONTSIZE,
    title_fontsize: int = TITLE_FONTSIZE,
    legend_fontsize: int = LEGEND_FONTSIZE,
    scatter_size: int = OVERLAY_SCATTER_SIZE,
    x_label: str = "Dim-1",
    y_label: str = "Dim-2",
    show_title: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)

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
            alpha=OVERLAY_ALPHA,
            size=scatter_size,
            show_legend=False,
        )

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel(x_label, fontsize=axis_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_fontsize)

    if show_title and title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.2)

    if show_legend:
        from matplotlib.lines import Line2D

        cmap = plt.get_cmap("tab10")
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color_map.get(label, cmap(idx % cmap.N)),
                label=label,
            )
            for idx, label in enumerate(instance_labels)
        ]

        ax.legend(
            handles=legend_handles,
            loc="best",
            fontsize=legend_fontsize,
            ncol=2,
            title="Instance",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI_SINGLE)
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
    *,
    show_legend: bool = True,
    axis_fontsize: int = AXIS_LABEL_FONTSIZE,
    tick_fontsize: int = TICK_LABEL_FONTSIZE,
    title_fontsize: int = TITLE_FONTSIZE,
    legend_fontsize: int = LEGEND_FONTSIZE,
    scatter_size: int = TRAJECTORY_SCATTER_SIZE,
    trajectory_point_size: int = TRAJECTORY_POINT_SIZE,
    trajectory_linewidth: float = TRAJECTORY_LINEWIDTH,
    x_label: str = "Dim-1",
    y_label: str = "Dim-2",
    show_title: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)

    all_coords = np.concatenate([coords_by_epoch[e] for e in epoch_labels], axis=0)
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    for sample_idx in trajectory_indices:
        pts = np.stack([coords_by_epoch[e][sample_idx] for e in epoch_labels], axis=0)
        inst_idx = sample_instance_indices[sample_idx]
        inst_label = instance_labels[inst_idx]

        cmap = plt.get_cmap("tab10")
        color = color_map.get(inst_label, cmap(inst_idx % cmap.N))

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            alpha=TRAJECTORY_LINE_ALPHA,
            linewidth=trajectory_linewidth,
        )
        ax.scatter(
            pts[0, 0],
            pts[0, 1],
            color=color,
            s=trajectory_point_size,
            alpha=TRAJECTORY_POINT_ALPHA,
        )
        ax.scatter(
            pts[-1, 0],
            pts[-1, 1],
            color=color,
            s=trajectory_point_size,
            alpha=TRAJECTORY_POINT_ALPHA,
        )

    for epoch in epoch_labels:
        coords = coords_by_epoch[epoch]

        _scatter_by_instance(
            ax,
            coords,
            sample_instance_indices,
            instance_labels,
            color_map,
            alpha=TRAJECTORY_ALPHA,
            size=scatter_size,
            show_legend=False,
        )

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel(x_label, fontsize=axis_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_fontsize)

    if show_title and title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.2)

    if show_legend:
        from matplotlib.lines import Line2D

        cmap = plt.get_cmap("tab10")
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=color_map.get(label, cmap(idx % cmap.N)),
                marker="o",
                linestyle="",
                label=label,
            )
            for idx, label in enumerate(instance_labels)
        ]

        ax.legend(
            handles=legend_handles,
            loc="best",
            fontsize=legend_fontsize,
            ncol=2,
            title="Instance",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI_SINGLE)
    plt.close(fig)

def save_coordinates_csv(
    coords_by_epoch: dict[str, np.ndarray],
    epoch_labels: Sequence[str],
    sample_indices: np.ndarray,
    instance_names: Sequence[str],
    instance_indices: np.ndarray,
    relative_paths: Sequence[str],
    filenames: Sequence[str],
    output_path: Path,
    method_name: str = "projection",
) -> None:
    rows = []
    col1 = f"{method_name.lower()}_1"
    col2 = f"{method_name.lower()}_2"

    for epoch in epoch_labels:
        coords = coords_by_epoch[epoch]
        for idx, (x, y) in enumerate(coords):
            rows.append(
                {
                    "sample_idx": int(sample_indices[idx]),
                    "subset_idx": idx,
                    "instance_idx": int(instance_indices[idx]),
                    "instance_name": instance_names[idx],
                    "filename": filenames[idx],
                    "relative_path": relative_paths[idx],
                    "epoch": epoch,
                    col1: float(x),
                    col2: float(y),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize representation evolution across training checkpoints using PCA projection."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Single dataset root with hierarchical structure (object_name/parameter_name/instance_*/).",
    )
    parser.add_argument(
        "--filename-col",
        type=str,
        default="image",
        help="Name of the CSV column that contains the image filename.",
    )
    parser.add_argument(
        "--instance-glob",
        type=str,
        default=DEFAULT_INSTANCE_GLOB,
        help="Glob used to discover instance folders under --dataset-root.",
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
        default=["action", "ssl"],
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
        help="Directory where per-parameter plots and CSV files will be written.",
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
        help="Optional per-instance cap. If > 0, samples are drawn evenly from each instance.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="This argument is deprecated and ignored. PCA is now applied directly to extract 2 components.",
    )
    parser.add_argument(
        "--use-pacmap",
        action="store_true",
        default=DEFAULT_USE_PACMAP,
        help="Use PACMAP instead of PCA for dimensionality reduction (default: False, uses PCA).",
    )
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=150,
        help="How many sample trajectories to draw in the development plot.",
    )
    parser.add_argument(
        "--include-untrained",
        action="store_true",
        default=True,
        help="Include an untrained model (weights initialized to 0) in the visualization (default: True).",
    )
    parser.add_argument(
        "--no-untrained",
        action="store_true",
        help="Exclude the untrained model from the visualization.",
    )
    parser.add_argument(
        "--supervised-model",
        type=Path,
        default=SUPERVISED_MODEL_PATH,
        help=(
            "Path to a supervised checkpoint file or a directory containing epoch checkpoints. "
            "If a directory is provided, all matching epochs are loaded."
        ),
    )
    return parser.parse_args()


def process_parameter(
    dataset_root: Path,
    object_name: str,
    parameter_name: str,
    args: argparse.Namespace,
    device: torch.device,
    color_map: dict[str, tuple[float, float, float, float]],
) -> None:
    """Process a single parameter for a specific object and create visualizations.
    
    For the given object_name/parameter_name combination, loads all instances 
    and creates plots for each model group.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {object_name} / {parameter_name}")
    print(f"{'='*80}")
    
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = ImageOnlyDataset(
        dataset_root,
        filename_col=args.filename_col,
        instance_glob=args.instance_glob,
        image_subdir=args.image_subdir,
        csv_name=args.csv_name,
        transform=transform,
        object_filter=object_name,
        parameter_filter=parameter_name,
    )

    if args.max_samples_per_instance > 0:
        subset_indices = select_balanced_subset_indices(dataset.instance_indices, args.max_samples_per_instance, args.seed)
    else:
        subset_indices = select_subset_indices(len(dataset), args.max_samples, args.seed)

    subset = Subset(dataset, subset_indices.tolist())
    filenames = [dataset.filenames[i] for i in subset_indices]
    relative_paths = [dataset.relative_paths[i] for i in subset_indices]
    sample_instance_indices = dataset.instance_indices[subset_indices]
    sample_instance_names = [dataset.instance_names[i] for i in subset_indices]
    instance_labels = dataset.instance_labels
    loader = build_loader(subset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Build a color map specific to this dataset's instance labels. Overwrite
    # the passed-in color_map to avoid missing-key errors when datasets have
    # different instance sets.
    color_map = build_instance_color_map(instance_labels)

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

        # Optionally extract embeddings from untrained model
        if args.include_untrained:
            print(f"[Load] Creating untrained model (weights initialized to 0)")
            try:
                untrained_model = create_untrained_model(checkpoint_paths[0], device)
                embeddings, _sample_indices = extract_embeddings(untrained_model, loader, device)
                if embeddings.shape[0] != len(subset_indices):
                    raise RuntimeError(
                        f"Embedding count mismatch for untrained model: got {embeddings.shape[0]}, expected {len(subset_indices)}"
                    )
                embeddings_by_epoch["untrained"] = embeddings
                print(f"[Done] untrained: embeddings shape = {embeddings.shape}")
            except Exception as exc:
                print(f"[Skip] Could not process untrained model: {exc}")

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
                f"Need at least two successful embeddings for group '{group_name}' to visualize training development."
            )

        # Construct epoch_labels with proper ordering (untrained first if present, then numerical epochs)
        epoch_labels = []
        if "untrained" in embeddings_by_epoch:
            epoch_labels.append("untrained")
        epoch_labels.extend([resolve_epoch_label(p) for p in used_checkpoints])

        coords_by_epoch, _, method_name = fit_common_embedding_projection(
            embeddings_by_epoch=embeddings_by_epoch,
            use_pacmap=args.use_pacmap,
            pca_components=args.pca_components if args.pca_components > 0 else None,
            seed=args.seed,
        )
        print(f"[Info] {method_name} fitting completed successfully.")

        # Create output directory with object and parameter names as subdirectories
        group_output_dir = args.output_dir / object_name / parameter_name / args.model_arch / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"pacmap_evolution_{args.model_arch}_{group_name}"
        grid_path = group_output_dir / f"{prefix}_grid.png"
        overlay_path = group_output_dir / f"{prefix}_overlay.png"
        traj_path = group_output_dir / f"{prefix}_trajectories.png"
        csv_path = group_output_dir / f"{prefix}_coordinates.csv"

        title = f"{method_name} projection of the last-layer representations ({args.model_arch} / {group_name})"
        
        try:
            print(f"[Plot] Creating epoch grid...")
            plot_epoch_grid(
                coords_by_epoch,
                epoch_labels,
                sample_instance_indices,
                instance_labels,
                color_map,
                grid_path,
                title,
            )
            print(f"[Plot] Epoch grid saved to {grid_path}")
        except Exception as e:
            print(f"[Error] Failed to create epoch grid: {e}")
            import traceback
            traceback.print_exc()

        try:
            print(f"[Plot] Creating overlay plot...")
            plot_overlay(
                coords_by_epoch,
                epoch_labels,
                sample_instance_indices,
                instance_labels,
                color_map,
                overlay_path,
                title + " — overlay",
            )
            print(f"[Plot] Overlay plot saved to {overlay_path}")
        except Exception as e:
            print(f"[Error] Failed to create overlay plot: {e}")
            import traceback
            traceback.print_exc()

        try:
            print(f"[Plot] Creating trajectory plot...")
            traj_n = min(args.trajectory_samples, len(subset_indices))
            traj_indices = np.sort(np.random.default_rng(args.seed).choice(len(subset_indices), size=traj_n, replace=False))
            plot_trajectories(
                coords_by_epoch,
                epoch_labels,
                sample_instance_indices,
                instance_labels,
                color_map,
                traj_path,
                title + " — sample trajectories",
                traj_indices,
            )
            print(f"[Plot] Trajectory plot saved to {traj_path}")
        except Exception as e:
            print(f"[Error] Failed to create trajectory plot: {e}")
            import traceback
            traceback.print_exc()

        try:
            print(f"[Plot] Saving coordinates CSV...")
            save_coordinates_csv(
                coords_by_epoch,
                epoch_labels,
                subset_indices,
                sample_instance_names,
                sample_instance_indices,
                relative_paths,
                filenames,
                csv_path,
                method_name=method_name,
            )
            print(f"[Plot] Coordinates CSV saved to {csv_path}")
        except Exception as e:
            print(f"[Error] Failed to save coordinates CSV: {e}")
            import traceback
            traceback.print_exc()

        print("[Saved]")
        print(f"  {grid_path}")
        print(f"  {overlay_path}")
        print(f"  {traj_path}")
        print(f"  {csv_path}")
        print(f"\n[Done] Group '{group_name}' completed successfully!\n")

    # Process supervised checkpoints (single file or full directory of epochs).
    supervised_checkpoints = []
    try:
        supervised_checkpoints = resolve_supervised_checkpoints(
            args.supervised_model,
            args.checkpoint_pattern,
            epochs_filter,
        )
    except Exception as exc:
        print(f"[Warning] Could not resolve supervised checkpoints: {exc}")

    if supervised_checkpoints:
        print(f"\n[Group] supervised")
        print(f"[Info] Supervised checkpoints: {[str(p) for p in supervised_checkpoints]}")

        embeddings_by_epoch: dict[str, np.ndarray] = {}
        used_checkpoints: list[Path] = []

        # Load untrained model for supervised using the first checkpoint as reference.
        if args.include_untrained:
            print(f"[Load] Creating untrained model (weights initialized to 0)")
            try:
                untrained_model = create_untrained_model(supervised_checkpoints[0], device)
                embeddings, _sample_indices = extract_embeddings(untrained_model, loader, device)
                if embeddings.shape[0] != len(subset_indices):
                    raise RuntimeError(
                        f"Embedding count mismatch for untrained model: got {embeddings.shape[0]}, expected {len(subset_indices)}"
                    )
                embeddings_by_epoch["untrained"] = embeddings
                print(f"[Done] untrained: embeddings shape = {embeddings.shape}")
            except Exception as exc:
                print(f"[Skip] Could not process untrained model: {exc}")

        for ckpt_path in supervised_checkpoints:
            epoch_label = resolve_epoch_label(ckpt_path)
            print(f"[Load] {ckpt_path.name}")
            try:
                model, _ = load_mv_model(str(ckpt_path), device=str(device))
                embeddings, _sample_indices = extract_embeddings(model, loader, device)
                if embeddings.shape[0] != len(subset_indices):
                    raise RuntimeError(
                        f"Embedding count mismatch for supervised checkpoint {ckpt_path.name}: got {embeddings.shape[0]}, expected {len(subset_indices)}"
                    )
                embeddings_by_epoch[epoch_label] = embeddings
                used_checkpoints.append(ckpt_path)
                print(f"[Done] epoch {epoch_label}: embeddings shape = {embeddings.shape}")
            except Exception as exc:
                print(f"[Skip] Could not process supervised checkpoint {ckpt_path.name}: {exc}")

        if len(embeddings_by_epoch) < 2:
            print(f"[Warning] Need at least two embeddings for supervised model visualization. Skipping.")
        else:
            # Construct epoch_labels with proper ordering (untrained first if present, then loaded checkpoints).
            epoch_labels = []
            if "untrained" in embeddings_by_epoch:
                epoch_labels.append("untrained")
            epoch_labels.extend([resolve_epoch_label(p) for p in used_checkpoints])

            coords_by_epoch, _, method_name = fit_common_embedding_projection(
                embeddings_by_epoch=embeddings_by_epoch,
                use_pacmap=args.use_pacmap,
                pca_components=args.pca_components if args.pca_components > 0 else None,
                seed=args.seed,
            )
            print(f"[Info] {method_name} fitting completed successfully.")

            group_output_dir = args.output_dir / object_name / parameter_name / "supervised"
            group_output_dir.mkdir(parents=True, exist_ok=True)
            prefix = "pacmap_evolution_supervised"
            grid_path = group_output_dir / f"{prefix}_grid.png"
            overlay_path = group_output_dir / f"{prefix}_overlay.png"
            traj_path = group_output_dir / f"{prefix}_trajectories.png"
            csv_path = group_output_dir / f"{prefix}_coordinates.csv"

            title = f"{method_name} projection of the last-layer representations ({args.model_arch} / supervised)"
            
            try:
                print(f"[Plot] Creating epoch grid...")
                plot_epoch_grid(
                    coords_by_epoch,
                    epoch_labels,
                    sample_instance_indices,
                    instance_labels,
                    color_map,
                    grid_path,
                    title,
                )
                print(f"[Plot] Epoch grid saved to {grid_path}")
            except Exception as e:
                print(f"[Error] Failed to create epoch grid: {e}")
                import traceback
                traceback.print_exc()

            try:
                print(f"[Plot] Creating overlay plot...")
                plot_overlay(
                    coords_by_epoch,
                    epoch_labels,
                    sample_instance_indices,
                    instance_labels,
                    color_map,
                    overlay_path,
                    title + " — overlay",
                )
                print(f"[Plot] Overlay plot saved to {overlay_path}")
            except Exception as e:
                print(f"[Error] Failed to create overlay plot: {e}")
                import traceback
                traceback.print_exc()

            try:
                print(f"[Plot] Creating trajectory plot...")
                traj_n = min(args.trajectory_samples, len(subset_indices))
                traj_indices = np.sort(np.random.default_rng(args.seed).choice(len(subset_indices), size=traj_n, replace=False))
                plot_trajectories(
                    coords_by_epoch,
                    epoch_labels,
                    sample_instance_indices,
                    instance_labels,
                    color_map,
                    traj_path,
                    title + " — sample trajectories",
                    traj_indices,
                )
                print(f"[Plot] Trajectory plot saved to {traj_path}")
            except Exception as e:
                print(f"[Error] Failed to create trajectory plot: {e}")
                import traceback
                traceback.print_exc()

            try:
                print(f"[Plot] Saving coordinates CSV...")
                save_coordinates_csv(
                    coords_by_epoch,
                    epoch_labels,
                    subset_indices,
                    sample_instance_names,
                    sample_instance_indices,
                    relative_paths,
                    filenames,
                    csv_path,
                    method_name=method_name,
                )
                print(f"[Plot] Coordinates CSV saved to {csv_path}")
            except Exception as e:
                print(f"[Error] Failed to save coordinates CSV: {e}")
                import traceback
                traceback.print_exc()

            print("[Saved]")
            print(f"  {grid_path}")
            print(f"  {overlay_path}")
            print(f"  {traj_path}")
            print(f"  {csv_path}")
            print(f"\n[Done] Supervised model visualization completed successfully!\n")


def main() -> None:
    args = parse_args()
    
    # Handle the no-untrained flag
    if args.no_untrained:
        args.include_untrained = False
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Using device: {device}")
    print(f"[Info] Model root: {args.model_root / args.model_arch}")
    print(f"[Info] Model groups: {args.model_groups}")
    print(f"[Info] Include untrained model: {args.include_untrained}")
    print(f"[Info] Dimensionality reduction method: {'PACMAP' if args.use_pacmap else 'PCA'}")
    if args.supervised_model:
        print(f"[Info] Supervised model: {args.supervised_model}")
    
    print(f"[Info] Dataset root: {args.dataset_root}")
    
    # Collect all object names from the dataset
    object_names = collect_object_names(args.dataset_root)
    print(f"[Info] Found {len(object_names)} objects: {object_names}")
    
    if not object_names:
        print(f"[Warning] No objects found in dataset root: {args.dataset_root}")
        return
    
    # Build color map (using first object/parameter combination for consistency)
    transform_temp = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    # Get first parameter from first object for color map
    first_object = object_names[0]
    first_parameters = collect_parameter_names_for_object(args.dataset_root, first_object)
    
    try:
        if first_parameters:
            dataset_temp = ImageOnlyDataset(
                args.dataset_root,
                filename_col=args.filename_col,
                instance_glob=args.instance_glob,
                image_subdir=args.image_subdir,
                csv_name=args.csv_name,
                transform=transform_temp,
                object_filter=first_object,
                parameter_filter=first_parameters[0],
            )
            instance_labels = dataset_temp.instance_labels
        else:
            instance_labels = []
    except Exception as e:
        print(f"[Warning] Could not load instance labels from first object/parameter: {e}")
        instance_labels = []
    
    color_map = build_instance_color_map(instance_labels)

    # Process each object, then each parameter for that object
    for object_name in object_names:
        print(f"\n{'#'*80}")
        print(f"## Object: {object_name}")
        print(f"{'#'*80}")
        
        parameter_names = collect_parameter_names_for_object(args.dataset_root, object_name)
        print(f"[Info] Parameters for {object_name}: {parameter_names}")
        
        for parameter_name in parameter_names:
            try:
                process_parameter(args.dataset_root, object_name, parameter_name, args, device, color_map)
            except Exception as e:
                print(f"\n[Error] Failed to process {object_name}/{parameter_name}: {e}")
                import traceback
                traceback.print_exc()
                continue




if __name__ == "__main__":
    try:
        main()
        print("\n[Success] All processing completed successfully!")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        import traceback
        traceback.print_exc()
