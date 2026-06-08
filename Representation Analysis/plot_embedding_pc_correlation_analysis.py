"""
embedding_pc_correlation_analysis.py
────────────────────────────────────────────────────────────────────────────
Plot embedding–factor alignment and explained variance.

For every (dataset, factor, model) combination this script:
  1. Extracts L2-normalised backbone embeddings for rendered images.
  2. Runs PCA on the embeddings.
  3. Computes Spearman correlations between the transformation parameter and
     PC1 / PC2 scores.
  4. Plots |Spearman rho| for PC1 and PC2 as grouped bars.
  5. Plots PC1+PC2 explained variance ratio (EVR total) as a dashed line.
  6. Saves a CSV with the raw rho, p-values, EVR values, and sample counts.

This replaces the previous distance-vs-parameter-change plots.

Usage
─────
Edit the CONFIG section at the bottom and run:
    python embedding_pc_correlation_analysis.py
"""

import os
import re
from glob import glob
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # safe non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - only used when scipy is unavailable
    spearmanr = None

# Model checkpoints root (edit here to change all model paths).
MODEL_PATH_ROOT = r"C:\Users\silas\PycharmProjects\SimClr_MT"


# ─────────────────────────────────────────────────────────────────────────────
# 1) Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _get_backbone(model_path: str, device: str, use_sup_lin_projector: bool = False):
    """
    Return (backbone, preprocess) for a checkpoint path or ImageNet shorthand.

    The backbone returns raw pooled features because the classification head is
    replaced with Identity. This keeps embedding dimensionalities comparable.
    """
    import torch.nn as nn
    import torchvision
    from torchvision.models import ResNet50_Weights

    if model_path in ("V1", "V2"):
        if model_path == "V1":
            model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            prep = ResNet50_Weights.IMAGENET1K_V1.transforms()
        else:
            model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            prep = ResNet50_Weights.IMAGENET1K_V2.transforms()
    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        from pretrained.load_mvimgnet_model import load_mv_model
        model, prep = load_mv_model(
            model_path,
            device=device,
            use_sup_lin_projector=use_sup_lin_projector,
        )
    else:
        raise ValueError(f"Unknown model specifier: {model_path!r}")

    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model, prep


# ─────────────────────────────────────────────────────────────────────────────
# 2) Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

_IDX_RE = re.compile(r"(\d+)(?=\.(jpg|jpeg|png)$)", re.IGNORECASE)


def _sort_key(path: str) -> int:
    match = _IDX_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else 10**18


@torch.no_grad()
def extract_embeddings(
    images_dir: str,
    csv_path: str,
    model: torch.nn.Module,
    preprocess,
    device: str,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract L2-normalised embeddings and transformation-parameter values.

    The CSV is expected to contain at least one numeric parameter column. The
    last numeric column is used as the transformation parameter value.
    """
    raw = pd.read_csv(csv_path, header=None)
    try:
        float(raw.iloc[0, -1])
        df = raw
    except (ValueError, TypeError):
        df = pd.read_csv(csv_path)

    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob(os.path.join(images_dir, pattern)))
    image_paths.sort(key=_sort_key)

    n_items = min(len(image_paths), len(df))
    if n_items == 0:
        raise ValueError(f"No aligned image/CSV rows found in {images_dir}")

    image_paths = image_paths[:n_items]
    df = df.iloc[:n_items].reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        last_col = df.columns[-1]
        df[last_col] = pd.to_numeric(
            df[last_col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        param_vals = df[last_col].to_numpy(dtype=float)
    else:
        param_vals = df[numeric_cols[-1]].to_numpy(dtype=float)

    embeddings = []
    for start in range(0, n_items, batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            batch_images.append(preprocess(image))

        x = torch.stack(batch_images).to(device)
        emb = model(x)
        if emb.ndim == 4:
            emb = emb.squeeze(-1).squeeze(-1)
        emb = F.normalize(emb, dim=1)
        embeddings.append(emb.cpu())

    return torch.cat(embeddings).numpy(), param_vals


# ─────────────────────────────────────────────────────────────────────────────
# 3) PCA and Spearman statistics
# ─────────────────────────────────────────────────────────────────────────────

def pca_scores_and_evr(embeddings: np.ndarray, n_components: int = 2):
    """
    Compute PCA scores and explained variance ratios using NumPy SVD.

    Returns
    -------
    scores : np.ndarray, shape (N, n_components)
        PC scores for each image.
    evr : np.ndarray, shape (n_components,)
        Explained variance ratio for each requested component.
    """
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError(f"PCA needs an (N, D) matrix with N >= 3, got {x.shape}")

    finite_rows = np.isfinite(x).all(axis=1)
    x = x[finite_rows]
    if x.shape[0] < 3:
        raise ValueError("Fewer than 3 finite embeddings remain after filtering.")

    x_centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)

    n_available = min(n_components, vt.shape[0])
    components = vt[:n_available]
    scores = x_centered @ components.T

    eigenvalues = (singular_values**2) / max(x.shape[0] - 1, 1)
    total_variance = eigenvalues.sum()
    if total_variance <= 0:
        evr_full = np.zeros_like(eigenvalues)
    else:
        evr_full = eigenvalues / total_variance

    # Pad if fewer than two components are available.
    if n_available < n_components:
        scores = np.pad(scores, ((0, 0), (0, n_components - n_available)))
        evr = np.pad(evr_full[:n_available], (0, n_components - n_available))
    else:
        evr = evr_full[:n_components]

    return scores, evr


def spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return Spearman rho and p-value, filtering NaN/Inf values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")

    if spearmanr is not None:
        rho, p_value = spearmanr(x, y)
        return float(rho), float(p_value)

    # Fallback without p-value when scipy is unavailable.
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    rho = np.corrcoef(x_rank, y_rank)[0, 1]
    return float(rho), float("nan")


def _instance_dirs(root_dir: str) -> list[str]:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Factor directory does not exist: {root_dir}")
    return sorted(
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )


def _select_instance_dirs(instance_dirs: list[str], instance_mode: str) -> list[str]:
    """
    Choose which instance folders to analyse.

    instance_mode:
      - "last": use only the last sorted instance folder; matches figures titled
        "last instance".
      - "first": use only the first sorted instance folder.
      - "all": concatenate all instance folders before PCA.
      - "train": use only instances 1-4.
      - "test": use only instance 5.
    """
    if not instance_dirs:
        return []

    def _instance_index(path: str) -> Optional[int]:
        name = os.path.basename(path)
        match = re.search(r"instance[_-]?(\d+)", name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else None

    if instance_mode == "last":
        return [instance_dirs[-1]]
    if instance_mode == "first":
        return [instance_dirs[0]]
    if instance_mode == "all":
        return instance_dirs
    if instance_mode == "train":
        return [d for d in instance_dirs if (_instance_index(d) or 0) in {1, 2, 3, 4}]
    if instance_mode == "test":
        return [d for d in instance_dirs if (_instance_index(d) or 0) == 5]

    raise ValueError("instance_mode must be one of: 'last', 'first', 'all'")


def analyse_factor_model(
    factor_name: str,
    root_dir: str,
    model: torch.nn.Module,
    preprocess,
    device: str,
    batch_size: int,
    instance_mode: str = "last",
) -> dict:
    """Analyse one factor for one model."""
    all_instance_dirs = _instance_dirs(root_dir)
    selected_dirs = _select_instance_dirs(all_instance_dirs, instance_mode)

    embeddings_list = []
    param_list = []
    used_instances = []

    for instance_dir in selected_dirs:
        images_dir = os.path.join(instance_dir, "images")
        csv_path = os.path.join(instance_dir, "parameters.csv")
        if not os.path.isdir(images_dir) or not os.path.isfile(csv_path):
            print(f"    Skipping incomplete instance: {instance_dir}")
            continue

        emb, params = extract_embeddings(
            images_dir=images_dir,
            csv_path=csv_path,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=batch_size,
        )
        embeddings_list.append(emb)
        param_list.append(params)
        used_instances.append(os.path.basename(instance_dir))

    if not embeddings_list:
        raise ValueError(f"No usable instances found for factor {factor_name!r}")

    embeddings = np.vstack(embeddings_list)
    params = np.concatenate(param_list)

    valid = np.isfinite(params)
    embeddings = embeddings[valid]
    params = params[valid]

    scores, evr = pca_scores_and_evr(embeddings, n_components=2)
    # pca_scores_and_evr may filter non-finite embedding rows internally. The
    # extraction pipeline should not create non-finite embeddings, so params and
    # scores should still align here.
    if len(scores) != len(params):
        raise ValueError(
            f"Internal length mismatch after PCA for {factor_name}: "
            f"scores={len(scores)}, params={len(params)}"
        )

    pc1_rho, pc1_p = spearman_corr(scores[:, 0], params)
    pc2_rho, pc2_p = spearman_corr(scores[:, 1], params)

    return {
        "factor": factor_name,
        "pc1_rho": pc1_rho,
        "pc1_p": pc1_p,
        "pc2_rho": pc2_rho,
        "pc2_p": pc2_p,
        "pc1_abs_rho": abs(pc1_rho) if np.isfinite(pc1_rho) else np.nan,
        "pc2_abs_rho": abs(pc2_rho) if np.isfinite(pc2_rho) else np.nan,
        "pc1_evr": float(evr[0]),
        "pc2_evr": float(evr[1]),
        "evr_total": float(evr[0] + evr[1]),
        "n_images": int(len(params)),
        "n_instances": int(len(used_instances)),
        "instances": ",".join(used_instances),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4) Plotting
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_ORDER = [
    "camera.distance",
    "camera.elevation",
    "camera.azimuth",
    "camera.roll",
    "background.hue",
    "background.saturation",
    "background.value",
    "background.noise",
    "light.power",
    "light.azimuth",
    "light.elevation",
]

SHORT_FACTOR_LABELS = {
    "camera.distance": "cam dist",
    "camera.elevation": "cam elev",
    "camera.azimuth": "cam azim",
    "camera.roll": "cam roll",
    "background.hue": "bg hue",
    "background.saturation": "bg sat",
    "background.value": "bg value",
    "background.noise": "bg noise",
    "light.power": "light power",
    "light.azimuth": "light azim",
    "light.elevation": "light elev",
}

FULL_FACTOR_LABELS = {
    "camera.distance": "Camera Distance",
    "camera.elevation": "Camera Elevation",
    "camera.azimuth": "Camera Azimuth",
    "camera.roll": "Camera Roll",
    "background.hue": "Background Hue",
    "background.saturation": "Background Saturation",
    "background.value": "Background Value",
    "background.noise": "Background Noise",
    "light.power": "Light Power",
    "light.azimuth": "Light Azimuth",
    "light.elevation": "Light Elevation",
}


def _normalize_factor_name(name: str) -> str:
    return name.strip().lower().replace("_", ".")


def _label_for_factor(name: str, use_short_labels: bool) -> str:
    norm = _normalize_factor_name(name)
    if use_short_labels:
        return SHORT_FACTOR_LABELS.get(norm, FULL_FACTOR_LABELS.get(norm, name))
    return FULL_FACTOR_LABELS.get(norm, name)


def _ordered_factor_names(factor_root_dirs: dict[str, str]) -> list[str]:
    skip = {
        "camera.azimuth",
        "background.hue",
        "light.azimuth",
        "camera.elevation",
        "light.elevation",
        "camera.roll",
    }
    def _norm(name: str) -> str:
        return _normalize_factor_name(name)

    skip_norm = {_norm(name) for name in skip}

    # Keep the first seen original name for each normalized key.
    norm_to_name = {}
    for name in factor_root_dirs:
        norm = _norm(name)
        if norm not in norm_to_name:
            norm_to_name[norm] = name

    ordered = []
    for name in FACTOR_ORDER:
        norm = _norm(name)
        if norm in skip_norm:
            continue
        key = norm_to_name.get(norm)
        if key is not None:
            ordered.append(key)

    ordered_norm = {_norm(name) for name in ordered}
    extras = sorted(
        name for name in factor_root_dirs
        if _norm(name) not in skip_norm and _norm(name) not in ordered_norm
    )
    return ordered + extras


def _discover_factor_dirs(object_root: str) -> dict[str, str]:
    """Return factor_name -> path for one object directory."""
    if not os.path.isdir(object_root):
        return {}

    factor_dirs = {}
    for name in sorted(os.listdir(object_root)):
        path = os.path.join(object_root, name)
        if not os.path.isdir(path):
            continue

        # Keep factors that contain at least one instance folder.
        has_instance = any(
            os.path.isdir(os.path.join(path, child))
            for child in os.listdir(path)
        )
        if has_instance:
            factor_dirs[name] = path

    return factor_dirs


def _discover_object_dirs(dataset_root: str) -> list[str]:
    if not os.path.isdir(dataset_root):
        return []
    return [
        name
        for name in sorted(os.listdir(dataset_root))
        if os.path.isdir(os.path.join(dataset_root, name))
    ]


def _add_sig_star(ax, x_pos: float, value: float, p_value: float, alpha: float = 0.05):
    if not np.isfinite(value):
        return
    mark = _star(p_value, alpha)
    if not mark:
        return
    ax.text(
        x_pos,
        min(value + 0.035, 1.045),
        mark,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _star(p_value: float, alpha: float = 0.05) -> str:
    if np.isfinite(p_value) and p_value < alpha:
        return "*"
    return ""


def plot_pc_correlation_figure(
    results_df: pd.DataFrame,
    dataset_name: str,
    model_labels: list[str],
    factor_names: list[str],
    output_path: str,
    use_short_labels: bool = False,
    use_secondary_evr_axis: bool = False,
    alpha: float = 00.05,
):
    """Create the stacked model figure."""
    if results_df.empty:
        raise ValueError("Cannot plot an empty results DataFrame.")

    n_models = len(model_labels)
    n_factors = len(factor_names)
    fig_height = max(3.2 * n_models, 5.5)
    fig_width = max(12.0, 0.85 * n_factors + 5.0)

    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=not use_secondary_evr_axis,
        constrained_layout=False,
    )
    if n_models == 1:
        axes = [axes]

    x = np.arange(n_factors)
    width = 0.34

    pc1_color = "#4C72B0"
    pc2_color = "#DD8452"
    evr_color = "#55A868"

    model_title_map = {
        "action_ssl": "Action",
        "action": "Action",
        "ssl": "SSL",
        "supervised": "Supervised",
    }

    def _display_title(raw_label: str) -> str:
        lowered = raw_label.strip().lower()
        for key, label in model_title_map.items():
            if key in lowered:
                return label
        return raw_label

    for ax, model_label in zip(axes, model_labels):
        sub = results_df[results_df["model"] == model_label].set_index("factor")
        sub = sub.reindex(factor_names)

        pc1 = sub["pc1_abs_rho"].to_numpy(dtype=float)
        pc2 = sub["pc2_abs_rho"].to_numpy(dtype=float)
        pc1_p = sub["pc1_p"].to_numpy(dtype=float)
        pc2_p = sub["pc2_p"].to_numpy(dtype=float)
        evr_total = sub["evr_total"].to_numpy(dtype=float)

        ax.bar(
            x - width / 2,
            pc1,
            width,
            color=pc1_color,
            edgecolor="black",
            linewidth=0.4,
            label="PC1 |ρ|",
        )
        ax.bar(
            x + width / 2,
            pc2,
            width,
            color=pc2_color,
            edgecolor="black",
            linewidth=0.4,
            label="PC2 |ρ|",
        )

        # Significance stars removed per request.
        # for xpos, val, pval in zip(x - width / 2, pc1, pc1_p):
        #     _add_sig_star(ax, xpos, val, pval, alpha=alpha)
        # for xpos, val, pval in zip(x + width / 2, pc2, pc2_p):
        #     _add_sig_star(ax, xpos, val, pval, alpha=alpha)

        if use_secondary_evr_axis:
            evr_ax = ax.twinx()
            evr_ax.plot(
                x,
                evr_total,
                color=evr_color,
                marker="o",
                markersize=4,
                linestyle="--",
                linewidth=1.4,
                label="EVR total",
            )
            evr_ax.set_ylim(0.0, 1.08)
            evr_ax.set_ylabel("Explained variance total", color=evr_color, fontsize=10)
            evr_ax.tick_params(axis="y", colors=evr_color, labelsize=9)
        else:
            ax.plot(
                x,
                evr_total,
                color=evr_color,
                marker="o",
                markersize=4,
                linestyle="--",
                linewidth=1.4,
                label="EVR total",
            )

        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("")
        ax.set_title(_display_title(model_label), fontsize=11, fontweight="bold", pad=5)
        ax.grid(axis="y", alpha=0.25)

    labels = [_label_for_factor(f, use_short_labels) for f in factor_names]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right", fontsize=9)

    # Single shared y-axis label on the left.
    fig.text(0.015, 0.5, "|Spearmon ρ| / Explained Variance Total", va="center", rotation="vertical", fontsize=10)

    handles = [
        Patch(facecolor=pc1_color, edgecolor="black", label="PC1 |ρ|"),
        Patch(facecolor=pc2_color, edgecolor="black", label="PC2 |ρ|"),
        Line2D([0], [0], color=evr_color, marker="o", linestyle="--", label="PC1+PC2 EVR"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )


    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.915))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_pc_correlation_analysis(
    models: list[str],
    model_labels: list[str],
    factor_root_dirs: dict[str, str],
    dataset_name: str,
    output_dir: str,
    device: Optional[str] = None,
    batch_size: int = 64,
    use_sup_lin_projector: bool = False,
    instance_mode: str = "last",
    use_short_labels: bool = False,
    use_secondary_evr_axis: bool = False,
    alpha: float = 0.05,
):
    """Run PCA/Spearman analysis for one dataset and save CSV + figure."""
    if len(models) != len(model_labels):
        raise ValueError("models and model_labels must have the same length.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    factor_names = _ordered_factor_names(factor_root_dirs)
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Instance mode: {instance_mode}")

    print("\nLoading models …")
    loaded_models = {}
    for model_path, model_label in zip(models, model_labels):
        print(f"  {model_label}")
        model, preprocess = _get_backbone(model_path, device, use_sup_lin_projector)
        loaded_models[model_label] = (model, preprocess)

    rows = []
    for factor_name in factor_names:
        root_dir = factor_root_dirs[factor_name]
        print(f"\n{'═' * 72}")
        print(f"Factor: {factor_name}")
        print(f"Root:   {root_dir}")

        for model_label in model_labels:
            model, preprocess = loaded_models[model_label]
            print(f"  [{model_label}]")
            stats = analyse_factor_model(
                factor_name=factor_name,
                root_dir=root_dir,
                model=model,
                preprocess=preprocess,
                device=device,
                batch_size=batch_size,
                instance_mode=instance_mode,
            )
            stats.update({"dataset": dataset_name, "model": model_label})
            rows.append(stats)
            print(
                f"    PC1 rho={stats['pc1_rho']:+.4f}, p={stats['pc1_p']:.2e}; "
                f"PC2 rho={stats['pc2_rho']:+.4f}, p={stats['pc2_p']:.2e}; "
                f"EVR total={stats['evr_total']:.4f}; N={stats['n_images']}"
            )

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"pc_correlation_{dataset_name}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV → {csv_path}")

    fig_path = os.path.join(output_dir, f"pc_correlation_{dataset_name}.png")
    plot_pc_correlation_figure(
        results_df=results_df,
        dataset_name=dataset_name,
        model_labels=model_labels,
        factor_names=factor_names,
        output_path=fig_path,
        use_short_labels=use_short_labels,
        use_secondary_evr_axis=use_secondary_evr_axis,
        alpha=alpha,
    )
    print(f"Saved figure → {fig_path}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit this section and run the script
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    _ROOT = r"C:\Users\silas\PycharmProjects\SimClr_MT"
    DATASET_ROOT = os.path.join(_ROOT, "dataset_one_transformation")

    # Order below matches the example figure: supervised, SSL, action-SSL.
    SSL_MODEL_PATH = r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\ssl\epoch_49.pt"
    ACTION_SSL_MODEL_PATH = r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\action\epoch_49.pt"
    SUPERVISED_MODEL_PATH = r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\supervised_MAPS_resnet18_seed0_1.pt"


    MODELS = [
        SUPERVISED_MODEL_PATH,
        SSL_MODEL_PATH,
        ACTION_SSL_MODEL_PATH,
    ]
    MODEL_LABELS = [
        "supervised_MAPS_V1",
        "ssl_resnet_18_MAPS",
        "action_ssl_resnet18_MAPS",
    ]

    USE_SUP_LIN_PROJECTOR = False

    # "last" reproduces figures that analyse the last instance only.
    # Use "all" to concatenate all instance folders before PCA.
    INSTANCE_MODE = "last"

    # Previous advice: because |ρ| and EVR are both in [0, 1], a single y-axis
    # is cleaner. Set this to True to reproduce the old right-axis style.
    USE_SECONDARY_EVR_AXIS = False

    # Set True for shorter x-axis labels such as "cam dist" and "bg sat".
    USE_SHORT_LABELS = False

    OUTPUT_DIR = os.path.join(_ROOT, "embedding_pc_correlation_plots")

    object_names = _discover_object_dirs(DATASET_ROOT)
    if not object_names:
        raise FileNotFoundError(f"No object folders found under {DATASET_ROOT}")

    for object_name in object_names:
        object_root = os.path.join(DATASET_ROOT, object_name)
        factor_root_dirs = _discover_factor_dirs(object_root)
        if not factor_root_dirs:
            print(f"Skipping {object_name}: no factor folders found.")
            continue

        for split_name in ("train", "test"):
            run_pc_correlation_analysis(
                models=MODELS,
                model_labels=MODEL_LABELS,
                factor_root_dirs=factor_root_dirs,
                dataset_name=f"{object_name}_{split_name}",
                output_dir=os.path.join(OUTPUT_DIR, object_name, split_name),
                device=None,
                batch_size=64,
                use_sup_lin_projector=USE_SUP_LIN_PROJECTOR,
                instance_mode=split_name,
                use_short_labels=USE_SHORT_LABELS,
                use_secondary_evr_axis=USE_SECONDARY_EVR_AXIS,
                alpha=0.05,
            )
