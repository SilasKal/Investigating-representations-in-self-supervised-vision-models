"""Fit one PCA on embeddings from multiple objects for one varying parameter.

Example use case:
- background hue variation for elephant, monitor, banana
- extract embeddings from one model/layer
- fit a single PCA over all samples
- plot PC1/PC2 colored by object category
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
except Exception:
    PACMAP_AVAILABLE = False

from pretrained.load_mvimgnet_model import load_mv_model

DEFAULT_DATASET_ROOT = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_one_transformation")
DEFAULT_MODEL_ROOT = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files")
DEFAULT_MODEL_ARCH = "resnet18"
DEFAULT_MODEL_TYPES = ["action", "ssl", "supervised"]
DEFAULT_LAYER = "layer4"
DEFAULT_EPOCH = 49
DEFAULT_SUPERVISED_MODEL = Path(
    r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\supervised_MAPS_resnet18_seed0_1.pt"
)
DEFAULT_USE_PACMAP = False

# Plot style defaults
DEFAULT_AXIS_LABEL_FONTSIZE = 30
DEFAULT_TICK_LABEL_FONTSIZE = 24
DEFAULT_TITLE_FONTSIZE = 28
DEFAULT_LEGEND_FONTSIZE = 12
DEFAULT_LEGEND_TITLE_FONTSIZE = 13
DEFAULT_LEGEND_MARKER_SIZE = 8
DEFAULT_LEGEND_NCOL = 4
DEFAULT_SHOW_LEGEND_IN_PLOT = False
DEFAULT_DPI = 220
DEFAULT_LEGEND_DPI = 1200


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else math.inf, path.name)


def normalize_param_dir_name(name: str) -> str:
    return name.replace(".", "_").strip()


def save_legend_only(
    handles: list,
    labels: list[str],
    output_path: Path,
    title: str = "Object",
    fontsize: int = DEFAULT_LEGEND_FONTSIZE,
    title_fontsize: int = DEFAULT_LEGEND_TITLE_FONTSIZE,
    ncol: int = DEFAULT_LEGEND_NCOL,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Save a standalone legend as a tightly cropped PNG."""
    if not handles or not labels:
        return

    ncol = max(1, min(ncol, len(labels)))
    nrows = math.ceil(len(labels) / ncol)

    # The figure size is only a starting point; bbox_inches='tight' crops it closely.
    fig_width = max(4.0, 2.2 * ncol)
    fig_height = max(1.0, 0.45 * nrows + 0.6)
    fig = plt.figure(figsize=(fig_width, fig_height))

    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        title=title,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def collect_parameter_dirs(dataset_root: Path, objects: list[str] | None = None) -> list[str]:
    """Collect all parameter folder names that exist under the dataset root."""
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
    """Collect images from multiple objects for one parameter folder.

    Expected layout:
      dataset_root/
        object_name/
          parameter_name/
            instance_1/
              parameters.csv
              images/
            instance_2/
              ...

    Object category is derived from `object_name` and used for coloring.
    """

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


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        if not args.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        return args.checkpoint_path

    if args.model_type == "supervised":
        if not args.supervised_model.exists():
            raise FileNotFoundError(f"Supervised checkpoint not found: {args.supervised_model}")
        return args.supervised_model

    checkpoint_dir = args.model_root / args.model_arch / args.model_type
    checkpoints = sorted(checkpoint_dir.glob(args.checkpoint_pattern), key=lambda p: natural_key(p))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} with pattern {args.checkpoint_pattern}")

    if args.epoch is None:
        return checkpoints[-1]

    pattern = re.compile(rf"epoch_{args.epoch}(?:\D|$)")
    for ckpt in checkpoints:
        if pattern.search(ckpt.stem):
            return ckpt

    raise FileNotFoundError(f"No checkpoint for epoch {args.epoch} in {checkpoint_dir}")


def get_module_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if layer_name in modules:
        return modules[layer_name]
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    names_preview = ", ".join(sorted([name for name in modules if name])[:25])
    raise ValueError(f"Layer '{layer_name}' not found. First modules: {names_preview}")


def extract_layer_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    all_object_idx: list[torch.Tensor] = []
    all_sample_idx: list[torch.Tensor] = []

    try:
        with torch.no_grad():
            for images, object_idx, sample_idx in loader:
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
                all_object_idx.append(object_idx.detach().cpu())
                all_sample_idx.append(sample_idx.detach().cpu())
    finally:
        if hook is not None:
            hook.remove()

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    object_indices = torch.cat(all_object_idx, dim=0).numpy()
    sample_indices = torch.cat(all_sample_idx, dim=0).numpy()
    return embeddings, object_indices, sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one PCA over multiple objects for one changing parameter and color by object category."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--parameter-dir",
        type=str,
        default=None,
        help="Parameter folder name, e.g. 'background_hue' or 'background.hue'. If omitted, all parameters are processed.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Optional subset of object folder names. If omitted, all objects with that parameter are used.",
    )
    parser.add_argument("--filename-col", type=str, default="image")
    parser.add_argument(
        "--model-types",
        nargs="*",
        choices=["action", "ssl", "supervised"],
        default=DEFAULT_MODEL_TYPES,
        help="Model types to analyze. Default: action ssl supervised.",
    )
    parser.add_argument("--layer", type=str, default=DEFAULT_LAYER, help="Default uses the last layer before classification (avgpool for ResNet).")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-arch", type=str, default=DEFAULT_MODEL_ARCH)
    parser.add_argument("--checkpoint-pattern", type=str, default="epoch_*.pt")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH, help="Default epoch for action/ssl models (49).")
    parser.add_argument("--supervised-model", type=Path, default=DEFAULT_SUPERVISED_MODEL)
    parser.add_argument(
        "--use-pacmap",
        action="store_true",
        default=DEFAULT_USE_PACMAP,
        help="Use PACMAP instead of PCA for dimensionality reduction (default: False).",
    )
    parser.add_argument("--instance-glob", type=str, default="*")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-samples-per-object", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("pca_multi_object_outputs"))

    # Plot style options
    parser.add_argument("--axis-label-fontsize", type=int, default=DEFAULT_AXIS_LABEL_FONTSIZE)
    parser.add_argument("--tick-label-fontsize", type=int, default=DEFAULT_TICK_LABEL_FONTSIZE)
    parser.add_argument("--title-fontsize", type=int, default=DEFAULT_TITLE_FONTSIZE)
    parser.add_argument("--legend-fontsize", type=int, default=DEFAULT_LEGEND_FONTSIZE)
    parser.add_argument("--legend-title-fontsize", type=int, default=DEFAULT_LEGEND_TITLE_FONTSIZE)
    parser.add_argument("--legend-marker-size", type=float, default=DEFAULT_LEGEND_MARKER_SIZE)
    parser.add_argument("--legend-ncol", type=int, default=DEFAULT_LEGEND_NCOL)
    parser.add_argument("--show-legend-in-plot", action="store_true", default=DEFAULT_SHOW_LEGEND_IN_PLOT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--legend-dpi", type=int, default=DEFAULT_LEGEND_DPI)
    parser.add_argument(
        "--plot-title",
        type=str,
        default=None,
        help="Optional plot title. If omitted, no title is shown.",
    )
    return parser.parse_args()


def choose_subset_indices(
    object_indices: np.ndarray,
    max_samples: int,
    max_samples_per_object: int,
    seed: int,
) -> np.ndarray:
    all_idx = np.arange(len(object_indices))
    rng = np.random.default_rng(seed)

    if max_samples_per_object and max_samples_per_object > 0:
        chunks: list[np.ndarray] = []
        for obj_id in np.unique(object_indices):
            obj_idx = np.flatnonzero(object_indices == obj_id)
            if len(obj_idx) > max_samples_per_object:
                obj_idx = np.sort(rng.choice(obj_idx, size=max_samples_per_object, replace=False))
            chunks.append(obj_idx)
        if not chunks:
            return all_idx
        all_idx = np.sort(np.concatenate(chunks))

    if max_samples and max_samples > 0 and max_samples < len(all_idx):
        all_idx = np.sort(rng.choice(all_idx, size=max_samples, replace=False))

    return all_idx


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parameter_dirs = [args.parameter_dir] if args.parameter_dir else collect_parameter_dirs(args.dataset_root, args.objects)
    if not parameter_dirs:
        raise RuntimeError(f"No parameter folders found under {args.dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Found {len(parameter_dirs)} parameter folders: {parameter_dirs}")

    for parameter_dir in parameter_dirs:
        checkpoint_cache: dict[str, Path] = {}
        print(f"\n{'='*90}")
        print(f"Processing parameter: {parameter_dir}")
        print(f"{'='*90}")

        dataset = MultiObjectParameterDataset(
            dataset_root=args.dataset_root,
            parameter_dir=parameter_dir,
            objects=args.objects,
            filename_col=args.filename_col,
            instance_glob=args.instance_glob,
        )

        subset_indices = choose_subset_indices(
            object_indices=dataset.object_indices,
            max_samples=args.max_samples,
            max_samples_per_object=args.max_samples_per_object,
            seed=args.seed,
        )

        subset = Subset(dataset, subset_indices.tolist()) if len(subset_indices) < len(dataset) else dataset

        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        for model_type in args.model_types:
            checkpoint_path = checkpoint_cache.get(model_type)
            if checkpoint_path is None:
                checkpoint_path = resolve_checkpoint(argparse.Namespace(**{**vars(args), "model_type": model_type}))
                checkpoint_cache[model_type] = checkpoint_path

            model, _ = load_mv_model(str(checkpoint_path), device=str(device))
            embeddings, subset_object_idx, subset_local_idx = extract_layer_embeddings(model, loader, device, args.layer)

            if isinstance(subset, Subset):
                sample_indices = subset_indices[subset_local_idx]
            else:
                sample_indices = subset_local_idx

            method_name = "PCA"
            if args.use_pacmap:
                if not PACMAP_AVAILABLE:
                    raise ImportError("PACMAP is not installed. Install it with: pip install pacmap")
                method_name = "PACMAP"
                # normalize per-sample for PACMAP
                embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
                reducer = pacmap.PaCMAP(
                    n_components=2,
                    n_neighbors=10,
                    MN_ratio=0.5,
                    FP_ratio=2.0,
                    random_state=args.seed,
                    verbose=False,
                )
                coords = reducer.fit_transform(embeddings_normalized)
                var1, var2 = 0.0, 0.0
            else:
                pca = PCA(n_components=2, random_state=args.seed)
                coords = pca.fit_transform(embeddings)
                var1, var2 = pca.explained_variance_ratio_

            safe_layer = args.layer.replace(".", "_")
            safe_param = normalize_param_dir_name(parameter_dir)
            ckpt_tag = checkpoint_path.stem.replace(" ", "_")

            run_output_dir = args.output_dir / args.dataset_root.name / safe_param / model_type
            run_output_dir.mkdir(parents=True, exist_ok=True)

            method_tag = method_name.lower()
            plot_path = run_output_dir / f"{method_tag}_multi_object_{model_type}_{safe_layer}_{safe_param}_{ckpt_tag}.png"
            legend_path = run_output_dir / f"{method_tag}_multi_object_{model_type}_{safe_layer}_{safe_param}_{ckpt_tag}_legend.png"
            csv_path = run_output_dir / f"{method_tag}_multi_object_{model_type}_{safe_layer}_{safe_param}_{ckpt_tag}.csv"

            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.get_cmap("tab20")

            legend_handles = []
            legend_labels = []

            for obj_id, obj_name in enumerate(dataset.object_labels):
                mask = subset_object_idx == obj_id
                if not np.any(mask):
                    continue
                color = cmap(obj_id % cmap.N)
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=10,
                    alpha=0.85,
                    color=color,
                    edgecolors="none",
                    label=obj_name,
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=color,
                        markeredgecolor="none",
                        markersize=args.legend_marker_size,
                    )
                )
                legend_labels.append(obj_name)

            # show explained variance ratio for PC1/PC2 in axis labels
            if method_name == "PCA":
                ax.set_xlabel(f"PC1 ({var1 * 100:.2f}% var)", fontsize=args.axis_label_fontsize)
                ax.set_ylabel(f"PC2 ({var2 * 100:.2f}% var)", fontsize=args.axis_label_fontsize)
            else:
                ax.set_xlabel("Dim-1", fontsize=args.axis_label_fontsize)
                ax.set_ylabel("Dim-2", fontsize=args.axis_label_fontsize)

            ax.tick_params(axis="both", which="major", labelsize=args.tick_label_fontsize)
            ax.tick_params(axis="both", which="minor", labelsize=args.tick_label_fontsize)

            if args.plot_title:
                ax.set_title(args.plot_title, fontsize=args.title_fontsize)

            ax.grid(True, alpha=0.2)

            if args.show_legend_in_plot:
                ax.legend(
                    loc="best",
                    fontsize=args.legend_fontsize,
                    title="Object",
                    title_fontsize=args.legend_title_fontsize,
                )

            save_legend_only(
                legend_handles,
                legend_labels,
                legend_path,
                title="Object",
                fontsize=args.legend_fontsize,
                title_fontsize=args.legend_title_fontsize,
                ncol=args.legend_ncol,
                dpi=args.legend_dpi,
            )

            fig.tight_layout()
            fig.savefig(plot_path, dpi=args.dpi)
            plt.close(fig)

            out_df = pd.DataFrame(
                {
                    "sample_idx": sample_indices.astype(int),
                    "object_idx": subset_object_idx.astype(int),
                    "object_name": [dataset.object_labels[int(i)] for i in subset_object_idx],
                    "instance_name": [dataset.instance_names[i] for i in sample_indices],
                    "filename": [dataset.filenames[i] for i in sample_indices],
                    "relative_path": [dataset.relative_paths[i] for i in sample_indices],
                    f"{method_tag}_1": coords[:, 0].astype(float),
                    f"{method_tag}_2": coords[:, 1].astype(float),
                }
            )
            out_df.to_csv(csv_path, index=False)

            print(f"[Info] Device: {device}")
            print(f"[Info] Checkpoint: {checkpoint_path}")
            print(f"[Info] Samples: {len(sample_indices)}, Embedding dim: {embeddings.shape[1]}")
            print(f"[Info] Objects: {dataset.object_labels}")
            print(f"[Info] PCA variance: PC1={var1:.4f}, PC2={var2:.4f}, total={var1 + var2:.4f}")
            print(f"[Saved] {plot_path}")
            print(f"[Saved] {legend_path}")
            print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()

