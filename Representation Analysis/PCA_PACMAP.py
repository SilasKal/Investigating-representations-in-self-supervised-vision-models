import pacmap

from pathlib import Path
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.models import ResNet50_Weights

from pretrained.load_mvimgnet_model import resnet50

try:
    from torchvision import transforms as T
    _HAS_TV = True
except Exception:
    _HAS_TV = False


class ImageParamDataset(Dataset):
    """
    Expects:
      root/
        parameters.csv
        images/
    CSV must include a filename column and one or more parameter columns.
    """

    def __init__(
        self,
        root: str | Path,
        filename_col: str = "filename",
        param_cols: list[str] | None = None,
        transform=None,
        category_col: str | None = None,
        keep_category: int | str | None = None,
        drop_missing_files: bool = True,
        drop_rows_with_nan_params: bool = True,
        decimal_comma: bool = True,   # convert "12,3" -> "12.3"
    ):
        self.root = Path(root)
        self.csv_path = self.root / "parameters.csv"
        self.img_dir = self.root / "images"
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {self.csv_path}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {self.img_dir}")

        df = pd.read_csv(self.csv_path)

        if filename_col not in df.columns:
            raise ValueError(f"CSV missing filename column '{filename_col}'. Found: {list(df.columns)}")

        # Auto-detect parameter columns if not provided:
        # (prefer numeric columns, but also allow object columns that *can* be parsed to numeric)
        if param_cols is None:
            exclude = {filename_col}
            if category_col is not None:
                exclude.add(category_col)
            # take all non-excluded columns as candidates, then later force numeric
            candidates = [c for c in df.columns if c not in exclude]
            if not candidates:
                raise ValueError("No candidate parameter columns found. Pass param_cols explicitly.")
            param_cols = candidates

        for c in param_cols:
            if c not in df.columns:
                raise ValueError(f"CSV missing param column '{c}'. Found: {list(df.columns)}")

        # Optional category filtering
        if category_col is not None and keep_category is not None:
            if category_col not in df.columns:
                raise ValueError(f"CSV missing category column '{category_col}'. Found: {list(df.columns)}")
            df = df[df[category_col] == keep_category].reset_index(drop=True)

        # ---- Force params to numeric (fixes numpy.object_ issue) ----
        for c in param_cols:
            if decimal_comma and df[c].dtype == object:
                # handle German decimal commas and strip whitespace
                df[c] = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)

            # convert to numeric; invalid parses become NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if drop_rows_with_nan_params:
            before = len(df)
            df = df.dropna(subset=param_cols).reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                print(f"[ImageParamDataset] Dropped {dropped} rows with NaN/non-numeric params in {param_cols}.")

        # Resolve filepaths
        paths = (self.img_dir / df[filename_col].astype(str)).tolist()

        if drop_missing_files:
            keep = [p.exists() for p in paths]
            if not all(keep):
                missing = sum(1 for k in keep if not k)
                print(f"[ImageParamDataset] Dropping {missing} rows with missing image files.")
            df = df[keep].reset_index(drop=True)
            paths = [p for p in paths if p.exists()]

        self.df = df
        self.paths = paths
        self.filename_col = filename_col
        self.param_cols = list(param_cols)
        self.category_col = category_col

        # Reasonable default image transform if none is provided
        if self.transform is None and _HAS_TV:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            # fallback minimal conversion to CHW float tensor in [0,1]
            arr = np.array(img, dtype=np.float32) / 255.0  # HWC
            img = torch.from_numpy(arr).permute(2, 0, 1)   # CHW

        # IMPORTANT: force float32 numpy array (prevents numpy.object_)
        vals = self.df.loc[idx, self.param_cols].to_numpy(dtype=np.float32)
        params = torch.from_numpy(vals)

        if self.category_col is not None:
            cat = self.df.loc[idx, self.category_col]
            return img, params, cat

        return img, params


def make_loader(
    root,
    batch_size=128,
    num_workers=4,
    shuffle=False,
    **dataset_kwargs
):
    ds = ImageParamDataset(root, **dataset_kwargs)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return ds, dl

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# params_col = "background.hue"  # change if your column is named differently
# ds, loader = make_loader(
#     root=f"/Users/silas/PycharmProjects/MT/banana_{params_col.replace('.', '_')}",
#     filename_col="image",      # change if your column is named differently
#     param_cols=[params_col],           # or ["angle"] etc.
#     transform=transform,
#     batch_size=256,
#     num_workers=0,
#     shuffle=False
# )

import numpy as np
import torch



from torchvision import models
# state = torch.load("/Users/silas/PycharmProjects/MT/resnet50_V1.pth", map_location="cpu")
# model.load_state_dict(state)

import torch
import torch.nn as nn

# def get_backbone(model: nn.Module) -> nn.Module:
#     # 1) common pattern in SSL codebases
#     if hasattr(model, "backbone"):
#         return model.backbone
#
#     # 2) torchvision ResNet-like: has .fc and outputs logits by default
#     # Replace fc with identity to return pooled features
#     if hasattr(model, "fc"):
#         m = model
#         m.fc = nn.Identity()
#         return m
#
#     # 3) fallback
#     return model
#
# model = get_backbone(model)
#
# all_emb = []
# all_angle = []
#
# # If you need to filter a category, set this:
# target_category = None  # e.g. 5, or None if loader already filtered
#
# with torch.no_grad():
#     for batch in loader:
#         # Adapt this unpacking to your dataset
#         # Common possibilities:
#         # images, angle = batch
#         # images, category, angle = batch
#         # images, category, instance, angle = batch
#         if len(batch) == 2:
#             images, angle = batch
#             category = None
#         elif len(batch) == 3:
#             images, category, angle = batch
#         else:
#             images, category, _, angle = batch  # ignore instance
#
#         if target_category is not None and category is not None:
#             mask = (category == target_category)
#             if mask.sum() == 0:
#                 continue
#             images = images[mask]
#             angle = angle[mask]
#
#         images = images.to(device, non_blocking=True)
#
#         emb = model(images)  # [B, D] (or sometimes [B, D, 1, 1])
#         if emb.ndim == 4:
#             emb = emb.squeeze(-1).squeeze(-1)
#
#         # Normalize helps for SSL geometry (cosine structure)
#         emb = torch.nn.functional.normalize(emb, dim=1)
#
#         all_emb.append(emb.cpu())
#         all_angle.append(angle.detach().cpu())
#
# embeddings = torch.cat(all_emb).numpy()
# angles = torch.cat(all_angle).numpy().astype(np.float32)
#
# print("Embeddings:", embeddings.shape, "Angles:", angles.shape)
#
#
# import pacmap
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#
# reducer = pacmap.PaCMAP(
#     n_components=3,
#     n_neighbors=10,
#     MN_ratio=0.5,
#     FP_ratio=2.0,
#     random_state=42
# )
#
# emb3 = reducer.fit_transform(embeddings)
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection="3d")
#
# sc = ax.scatter(emb3[:, 0], emb3[:, 1], emb3[:, 2], c=angles, s=8)
# plt.colorbar(sc, label=params_col)
# ax.set_title("3D PaCMAP colored by angle")
# plt.savefig(f"pacmap_3d_" + params_col + ".png", dpi=300)
# plt.show()
#
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#
def compute_r2(embeddings, target, alpha=1.0):
    """
    embeddings: (N, D)
    target: (N,) or (N,1)
    """
    if target.ndim == 2:
        target = target.squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, target, test_size=0.3, random_state=42
    )

    # Ridge is safer than pure LinearRegression for high-D embeddings
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return r2


def compute_linear_regression_r2(features, target):
    """
    features: (N, d)
    target: (N,) or (N,1)
    Returns test R² using a fixed train/test split.
    """
    if target.ndim == 2:
        target = target.squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return r2_score(y_test, y_pred)

# r2 = compute_r2(embeddings, angles)
# # save r2 in a text file
# with open(f"r2_{params_col.replace('.', '_')}.txt", "w") as f:
#     f.write(f"R² for predicting {params_col} from embeddings: {r2:.4f}\n")
# print(f"R² for predicting {params_col} from embeddings: {r2:.4f}")



import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
def extract_layer_embeddings(model, loader, device, layer_names=None):
    """Extract pooled embeddings for several ResNet layers and all targets in the loader."""
    model.eval()
    model.to(device)

    if layer_names is None:
        layer_names = ["layer1", "layer2", "layer3", "layer4", "avgpool"]

    available_layers = [name for name in layer_names if hasattr(model, name)]
    if not available_layers:
        raise ValueError(f"Model does not expose any of the requested layers: {layer_names}")

    activations = {}
    hooks = []

    def make_hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()

        return fn

    for name in available_layers:
        hooks.append(getattr(model, name).register_forward_hook(make_hook(name)))

    collected = {name: [] for name in available_layers}
    targets = []

    try:
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                batch_targets = batch[1]

                _ = model(images)

                for name in available_layers:
                    feat = activations[name]
                    if feat.ndim > 2:
                        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    feat = F.normalize(feat, dim=1)
                    collected[name].append(feat.cpu())

                targets.append(batch_targets.detach().cpu())
    finally:
        for hook in hooks:
            hook.remove()

    collected = {
        name: torch.cat(chunks, dim=0).numpy()
        for name, chunks in collected.items()
        if chunks
    }
    targets = torch.cat(targets, dim=0).numpy()

    return collected, targets


def plot_layer_projection(embeddings, color_values, layer_name, param_col, model_path, instance_suffix):
    """Create a side-by-side PCA/PaCMAP plot for one layer."""
    from pathlib import Path
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    output_dir = Path(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    emb2_pca = pca.fit_transform(embeddings)

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42,
    )
    emb2_pacmap = reducer.fit_transform(embeddings)

    var1, var2 = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(emb2_pca[:, 0], emb2_pca[:, 1], c=color_values, cmap="viridis", s=8)
    axes[0].set_title(f"{layer_name} — PCA")
    axes[0].set_xlabel(f"PC1 ({var1 * 100:.2f}% variance)")
    axes[0].set_ylabel(f"PC2 ({var2 * 100:.2f}% variance)")

    sc1 = axes[1].scatter(emb2_pacmap[:, 0], emb2_pacmap[:, 1], c=color_values, cmap="viridis", s=8)
    axes[1].set_title(f"{layer_name} — PaCMAP")
    axes[1].set_xlabel("PaCMAP-1")
    axes[1].set_ylabel("PaCMAP-2")

    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label(param_col, fontsize=11)

    fig.suptitle(
        f"Representation in {layer_name} ({instance_suffix.replace('_', ' ')})\n"
        f"PCA variance: PC1 {var1 * 100:.2f}% | PC2 {var2 * 100:.2f}%",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)

    save_path = output_dir / f"layerwise_pca_pacmap_{layer_name}_{param_col.replace('.', '_')}_{instance_suffix}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved layer-wise PCA/PaCMAP plot → {save_path}")
    return emb2_pca, (var1, var2)


def plot_layer_pacmap(embeddings, color_values, layer_name, param_col, model_path, instance_suffix):
    """Create and save a PaCMAP 2D plot for one layer (colored by color_values)."""
    from pathlib import Path
    import matplotlib.pyplot as plt

    output_dir = Path(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=10,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=42,
    )
    emb2_pacmap = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(emb2_pacmap[:, 0], emb2_pacmap[:, 1], c=color_values, cmap="viridis", s=8)
    ax.set_title(f"{layer_name} — PaCMAP")
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(param_col, fontsize=11)

    fig.tight_layout()
    save_path = output_dir / f"layerwise_pacmap_{layer_name}_{param_col.replace('.', '_')}_{instance_suffix}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved layer-wise PaCMAP plot → {save_path}")
    return emb2_pacmap


def plot_layers_pacmap_grid(layer_embeddings_dict, color_values, layers, param_col, model_path, instance_suffix, layout='row'):
    """Create a combined PaCMAP figure for multiple layers.

    Parameters:
      layer_embeddings_dict: dict layer_name -> np.ndarray (N,D)
      layers: list of layer names in the order to plot
      layout: 'row' to plot all panels side-by-side, 'grid' for a 2-column layout
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import math

    output_dir = Path(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(layers)
    if n == 0:
        raise ValueError("No layers provided for plotting")

    if layout == 'row':
        rows = 1
        cols = n
    else:
        cols = 2
        rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(max(6, cols * 5), max(4, rows * 4)))

    # normalize axes into 2D numpy array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array(axes).reshape(rows, 1)
    else:
        axes = np.array(axes)

    vmin = float(np.min(color_values))
    vmax = float(np.max(color_values))
    sc = None

    for idx, layer_name in enumerate(layers):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        if layer_name not in layer_embeddings_dict:
            ax.set_visible(False)
            continue

        emb = layer_embeddings_dict[layer_name]
        if emb.shape[0] < 2:
            ax.set_visible(False)
            continue

        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
        emb2 = reducer.fit_transform(emb)

        sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=color_values, cmap="viridis", s=8, vmin=vmin, vmax=vmax)
        ax.set_title(f"{layer_name} — PaCMAP")
        ax.set_xlabel("PaCMAP-1")
        ax.set_ylabel("PaCMAP-2")

    # hide any unused subplots
    total_plots = rows * cols
    for idx in range(n, total_plots):
        r = idx // cols
        c = idx % cols
        try:
            axes[r, c].set_visible(False)
        except Exception:
            pass

    if sc is not None:
        # Place the colorbar to the right of the last plotted subplot
        try:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            last_idx = n - 1
            lr = last_idx // cols
            lc = last_idx % cols
            last_ax = axes[lr, lc]
            divider = make_axes_locatable(last_ax)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label(param_col, fontsize=11)
        except Exception:
            # fallback: shared colorbar for all axes (may appear at the right of the figure)
            cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
            cbar.set_label(param_col, fontsize=11)

    fig.suptitle(f"Layer-wise PaCMAP ({instance_suffix.replace('_', ' ')})", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    save_path = output_dir / f"layerwise_pacmap_grid_{param_col.replace('.', '_')}_{instance_suffix}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved combined layer PaCMAP grid → {save_path}")
    return save_path
def compute_r2_cv(embeddings, target, alpha=1.0):
    if target.ndim == 2:
        target = target.squeeze()

    model = Ridge(alpha=alpha)
    scores = cross_val_score(
        model,
        embeddings,
        target,
        cv=5,
        scoring="r2"
    )

    return scores.mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# for i in ["background.hue", "background.saturation", "background.value", "camera.distance", "camera.elevation", "camera.azimuth", "camera.roll", "light.azimuth", "light.power", "light.elevation" ]:
#     params_col = i  # change if your column is named differently
#     ds, loader = make_loader(
#         root=f"/Users/silas/PycharmProjects/MT/banana_{params_col.replace('.', '_')}",
#         filename_col="image",      # change if your column is named differently
#         param_cols=[params_col],           # or ["angle"] etc.
#         transform=transform,
#         batch_size=256,
#         num_workers=0,
#         shuffle=False
#     )
#     layer_embeddings, curr_params = extract_layer_embeddings(model, loader, device)
#
#
#     results = {}
#
#     for layer_name, emb in layer_embeddings.items():
#         r2 = compute_r2(emb, curr_params)
#         results[layer_name] = r2
#         print(f"predicting {params_col}: {layer_name:10s} R² = {r2:.4f}")
#     import matplotlib.pyplot as plt
#
#     # Ensure correct layer order
#     layer_order = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
#
#     # Extract values in correct order
#     r2_values = [results[layer] for layer in layer_order if layer in results]
#
#     plt.figure(figsize=(6, 4))
#     plt.plot(layer_order[:len(r2_values)], r2_values, marker="o")
#     plt.xlabel("Layer")
#     plt.ylabel("R²")
#     plt.title(f"Layer-wise R² for predicting {params_col}")
#     plt.ylim(0, 1)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"r2_by_layer_{params_col.replace('.', '_')}.png", dpi=300)
#     plt.show()
from scipy.stats import spearmanr
from pretrained.load_mvimgnet_model import load_mv_model

# ─────────────────────────────────────────────────────────────────────────────
# Option: USE_ONLY_LAST_INSTANCE
#   True  -> PC alignment computed only on the last (test) instance
#   False -> PC alignment computed on all instances EXCEPT the last one (train instances)
#
# Optional override: SELECT_ALIGNMENT_INSTANCE_INDEX
#   None -> keep USE_ONLY_LAST_INSTANCE behavior
#   int  -> compute PC alignment only on this specific instance index
#           (supports negative indices, e.g. -1)
# ─────────────────────────────────────────────────────────────────────────────
USE_ONLY_LAST_INSTANCE = True
SELECT_ALIGNMENT_INSTANCE_INDEX: int | None = None
instance_suffix = "last_instance" if USE_ONLY_LAST_INSTANCE else "train_instances"
# Option: include avgpool in the combined figure (set False to plot only layer1-4)
PLOT_AVGPOOL = True

# Accumulate Spearman results across all (model, factor) combinations
# Layout: spearman_results[model_path][factor] =
#   {"PC1_rho", "PC1_p", "PC2_rho", "PC2_p", "EVR_PC1", "EVR_PC2", "EVR_TOTAL"}
spearman_results: dict = {}
linear_regression_results: dict = {}

# for i in ["camera_distance", "camera_elevation", "camera_azimuth",
#           "background_hue" , "light_power",]:
for i in ["camera_distance", "camera_elevation", "camera_azimuth", "camera_roll",
     "background_hue", "background_saturation", "background_value", "background_noise",
     "light_power", "light_azimuth", "light_elevation"]:
# for i in [""]:
    import os
    import matplotlib.pyplot as plt
    import pkg_resources
    # instance wise representation analysis (all images from an instance share the same label, e.g. instance ID)

    # for index, j in enumerate([r"C:\Users\silas\PycharmProjects\SimClr_MT\finetuning_MAPS_5_instances_all_params\epoch_49.pt",
    #           r"C:\Users\silas\PycharmProjects\SimClr_MT\training_from_scratch_MAPS_5_instances_all_params\epoch_49.pt",
    #           r"C:\Users\silas\PycharmProjects\SimClr_MT\supervised_MAPS\sup_resnet50.pt", "V1", "V2"]):
    for index, j in enumerate(
            ["supervised_MAPS"]):
        if j in ["V1", "V2"]:
                print(f"Using torchvision ResNet50 {j} as backbone")
                if j == "V1":
                    model_path = "V1"
                    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                else:
                    model_path = "V2"
                    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                preprocess = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
        else:
            print(j)
            if index == 0:
                model_path = "finetuning_MAPS_5_instances_all_params"
            elif index == 1:
                model_path = "training_from_scratch_MAPS_5_instances_all_params"
            else:
                model_path = "supervised_MAPS_V1"
            # model, preprocess = load_mv_model(
            #     j,
            #     device='cpu')
            # print("Loading supervised Maps Model V1")
            # model_path = "ssl_resnet_18_MAPS"
            # model, preprocess = load_mv_model(
            #     r"C:\Users\silas\PycharmProjects\SimClr_MT\ssl_resnet_18_MAPS\epoch_49.pt",
            #     device='cpu')
            # model_path = "action_ssl_resnet18_MAPS"
            # model, preprocess = load_mv_model(
            #     r"C:\Users\silas\PycharmProjects\SimClr_MT\action_ssl_resnet18_MAPS\epoch_49.pt",
            #     device='cpu')
            # model_path = "supervised_MAPS_V1"
            # model, preprocess = load_mv_model(
            #     r"C:\Users\silas\PycharmProjects\SimClr_MT\supervised_MAPS_V1\supervised_MAPS_resnet18_seed0.pt",
            #     device='cpu')
        # model_path = r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\supervised"
        # model, preprocess = load_mv_model(
        #     r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\supervised\supervised_MAPS_resnet18_seed0_1.pt",
        #     device='cpu')
        model_path = r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\ssl"
        model, preprocess = load_mv_model(
            r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\ssl\epoch_49.pt",
            device='cpu')
        #
        # model, preprocess = load_mv_model(r'C:\Users\silas\PycharmProjects\SimClr_MT\finetuning_MAPS_5_instances_all_params\epoch_49.pt', device='cpu')
        # model, preprocess = load_mv_model(r'C:\Users\silas\PycharmProjects\SimClr_MT\training_from_scratch_MAPS_5_instances_all_params\epoch_49.pt', device='cpu')
        # model, preprocess = load_mv_model(r'C:\Users\silas\PycharmProjects\SimClr_MT\supervised_MAPS\sup_resnet50.pt', device='cpu')

        # model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # print('model loaded')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        BASE_DIR = fr"C:\Users\silas\PycharmProjects\SimClr_MT\banana_5_instances_{i}" # adjust to your path
        # BASE_DIR = fr"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_new_multiple\umbrella" # adjust to your path
        print(f'{i}, base dir: {BASE_DIR}')
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance_folders = sorted([
            os.path.join(BASE_DIR, d)
            for d in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, d))
        ])

        print(f"Found {len(instance_folders)} instances")

        all_embeddings = []
        all_param_values = []
        all_instance_ids = []   # one entry per sample

        # map folder param name (underscore) back to CSV column name (dot)
        param_col = i.replace("_", ".", 1)  # e.g. "camera_distance" -> "camera.distance"

        # --------------------------------------
        # Backbone extraction (avgpool features)
        # --------------------------------------
        model.fc = torch.nn.Identity()
        model = model.to(DEVICE).eval()

        for inst_id, inst_path in enumerate(instance_folders):

            print(f"Processing instance {inst_id}: {inst_path}")

            ds, loader = make_loader(
                root=inst_path,
                filename_col="image",      # adjust if needed
                # param_cols=[param_col],    # load the varying parameter
                param_cols = [],
                transform=transform,
                batch_size=256,
                num_workers=0,
                shuffle=False
            )

            with torch.no_grad():
                for batch in loader:
                    images, params = batch[0], batch[1]

                    images = images.to(DEVICE)
                    emb = model(images)                     # (B, 2048)
                    emb = F.normalize(emb, dim=1)

                    batch_size_n = emb.shape[0]
                    all_embeddings.append(emb.cpu())
                    all_param_values.append(params.cpu())
                    all_instance_ids.extend([inst_id] * batch_size_n)

        # --------------------------------------
        # Concatenate
        # --------------------------------------
        embeddings = torch.cat(all_embeddings).numpy()
        param_values = torch.cat(all_param_values).numpy().squeeze()  # (N,)

        print("Total embeddings:", embeddings.shape)

        # --------------------------------------
        # PaCMAP 3D
        # --------------------------------------
        reducer = pacmap.PaCMAP(
            n_components=3,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42
        )

        emb3 = reducer.fit_transform(embeddings)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm
        import numpy as np
        instance_ids = np.array(all_instance_ids)   # shape (N,)
        unique_ids = list(range(len(instance_folders)))
        print(f"Unique instance IDs: {unique_ids}")
        num_instances = len(unique_ids)

        # choose discrete colormap
        cmap = cm.get_cmap("tab10", num_instances)

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        # plot each instance separately
        for k, inst_id in enumerate(unique_ids):
            mask = instance_ids == inst_id
            ax.scatter(
                emb3[mask, 0],
                emb3[mask, 1],
                emb3[mask, 2],
                color=cmap(k),
                s=8,
                label=f"Instance {inst_id}"
            )

        ax.set_title(f"3D PaCMAP — Instance-wise representation ({i})")
        ax.set_xlabel("PaCMAP-1")
        ax.set_ylabel("PaCMAP-2")
        ax.set_zlabel("PaCMAP-3")

        ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        # plt.savefig(f"{model_path}/pacmap_instance_wise_banana_{i}_3d.png", dpi=300)
        # plt.show()

        # --------------------------------------
        # PaCMAP 2D — Instance-wise representation
        # --------------------------------------
        reducer2d = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42
        )

        emb2_pacmap = reducer2d.fit_transform(embeddings)

        fig2d, ax2d = plt.subplots(figsize=(9, 8))

        for k, inst_id in enumerate(unique_ids):
            mask = instance_ids == inst_id
            ax2d.scatter(
                emb2_pacmap[mask, 0],
                emb2_pacmap[mask, 1],
                color=cmap(k),
                s=8,
                label=f"Instance {inst_id}"
            )

        ax2d.set_title(f"2D PaCMAP — Instance-wise representation ({i})")
        ax2d.set_xlabel("PaCMAP-1")
        ax2d.set_ylabel("PaCMAP-2")
        ax2d.legend(loc="best", fontsize=9)

        plt.tight_layout()
        # plt.savefig(f"{model_path}/pacmap_instance_wise_banana_{i}_2d.png", dpi=300)
        # plt.show()

        # --------------------------------------
        # PCA 2D — Instance-wise representation
        # --------------------------------------
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        emb2_pca = pca.fit_transform(embeddings)

        var1, var2 = pca.explained_variance_ratio_
        print(f"PCA — PC1: {var1*100:.2f}%  PC2: {var2*100:.2f}%  "
              f"Total: {(var1+var2)*100:.2f}% of variance explained")

        fig_pca, ax_pca = plt.subplots(figsize=(9, 8))

        for k, inst_id in enumerate(unique_ids):
            mask = instance_ids == inst_id
            ax_pca.scatter(
                emb2_pca[mask, 0],
                emb2_pca[mask, 1],
                color=cmap(k),
                s=8,
                label=f"Instance {inst_id}"
            )

        ax_pca.set_title(
            f"2D PCA — Instance-wise representation ({i})\n"
            f"Explained variance — PC1: {var1*100:.2f}%  PC2: {var2*100:.2f}%  "
            f"(Total: {(var1+var2)*100:.2f}%)"
        )
        ax_pca.set_xlabel(f"PC1 ({var1*100:.2f}% variance)")
        ax_pca.set_ylabel(f"PC2 ({var2*100:.2f}% variance)")
        ax_pca.legend(loc="best", fontsize=9)
        ax_pca.annotate(
            f"Explained variance\nPC1: {var1*100:.2f}%\nPC2: {var2*100:.2f}%\nTotal: {(var1+var2)*100:.2f}%",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

        plt.tight_layout()
        # plt.savefig(f"{model_path}/pca_instance_wise_banana_{i}_2d.png", dpi=300)
        # plt.show()
        print(f"Saved PCA and PaCMAP plots for instance-wise representation analysis in the path {model_path}/pca_instance_wise_strawberry_2d.png")

        # --------------------------------------
        # Layer-wise PCA/PaCMAP colored by transformation parameter
        # --------------------------------------
        layer_names = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
        all_param_values_p = []
        layerwise_embeddings = {name: [] for name in layer_names}

        # Determine which instances to process for alignment analysis
        if SELECT_ALIGNMENT_INSTANCE_INDEX is not None:
            selected_idx_raw = SELECT_ALIGNMENT_INSTANCE_INDEX
            assert selected_idx_raw is not None
            selected_idx = int(selected_idx_raw)
            if selected_idx < 0:
                selected_idx += len(instance_folders)
            if selected_idx < 0 or selected_idx >= len(instance_folders):
                raise ValueError(
                    f"SELECT_ALIGNMENT_INSTANCE_INDEX={SELECT_ALIGNMENT_INSTANCE_INDEX} is out of range "
                    f"for {len(instance_folders)} instances."
                )
            instances_to_process = [(selected_idx, instance_folders[selected_idx])]
            instance_suffix = f"instance_{selected_idx}"
            print(f"Computing layer-wise alignment only for selected instance [{selected_idx}]: "
                  f"{instance_folders[selected_idx]}")
        elif USE_ONLY_LAST_INSTANCE:
            instances_to_process = [(len(instance_folders) - 1, instance_folders[-1])]
            instance_suffix = "last_instance"
            print(f"Computing layer-wise alignment only for last instance (test instance): {instance_folders[-1]}")
        else:
            instances_to_process = list(enumerate(instance_folders[:-1]))
            instance_suffix = "train_instances"
            print(f"Computing layer-wise alignment for all training instances (excluding last): "
                  f"{len(instances_to_process)} instances")

        for inst_id, inst_path in instances_to_process:
            ds_p, loader_p = make_loader(
                root=inst_path,
                filename_col="image",
                param_cols=[param_col],
                transform=transform,
                batch_size=256,
                num_workers=0,
                shuffle=False,
            )

            inst_layers, params_p = extract_layer_embeddings(model, loader_p, DEVICE, layer_names=layer_names)

            for layer_name in layer_names:
                if layer_name in inst_layers:
                    layerwise_embeddings[layer_name].append(torch.from_numpy(inst_layers[layer_name]))

            all_param_values_p.append(torch.from_numpy(params_p.astype(np.float32)))

        layerwise_embeddings = {
            name: torch.cat(chunks, dim=0).numpy()
            for name, chunks in layerwise_embeddings.items()
            if chunks
        }
        param_values_p = torch.cat(all_param_values_p, dim=0).numpy().squeeze()

        # Produce one PCA/PaCMAP figure per layer
        # Create a single combined PaCMAP grid for layer1..layer4
        layers_to_show = ["layer1", "layer2", "layer3", "layer4"]
        if PLOT_AVGPOOL:
            layers_to_show.append("avgpool")

        layer_dict = {ln: layerwise_embeddings[ln] for ln in layers_to_show if ln in layerwise_embeddings}
        if layer_dict:
            # layout='row' will place all panels side-by-side
            plot_layers_pacmap_grid(layer_dict, param_values_p, layers_to_show, param_col, model_path, instance_suffix, layout='row')

        # Keep the existing correlation / regression summary on the final pooled embedding (if available)
        if "avgpool" in layerwise_embeddings:
            embeddings_p = layerwise_embeddings["avgpool"]

            pca_p = PCA(n_components=2, random_state=42)
            emb2_pca_p = pca_p.fit_transform(embeddings_p)

            var1_p, var2_p = pca_p.explained_variance_ratio_
            print(f"PCA (avgpool, param-colored) — PC1: {var1_p*100:.2f}%  PC2: {var2_p*100:.2f}%  "
                  f"Total: {(var1_p+var2_p)*100:.2f}% of variance explained")

            fig_pca_p, ax_pca_p = plt.subplots(figsize=(9, 8))

            sc_p = ax_pca_p.scatter(
                emb2_pca_p[:, 0],
                emb2_pca_p[:, 1],
                c=param_values_p,
                cmap="viridis",
                s=8,
            )

            cbar_p = fig_pca_p.colorbar(sc_p, ax=ax_pca_p)
            cbar_p.set_label(param_col, fontsize=11)

            instance_label = (
                f" (instance {instances_to_process[0][0]} only)"
                if SELECT_ALIGNMENT_INSTANCE_INDEX is not None
                else (" (last instance only)" if USE_ONLY_LAST_INSTANCE else " (train instances only)")
            )
            ax_pca_p.set_title(
                f"2D PCA — avgpool colored by {param_col}{instance_label}\n"
                f"Explained variance — PC1: {var1_p*100:.2f}%  PC2: {var2_p*100:.2f}%  "
                f"(Total: {(var1_p+var2_p)*100:.2f}%)"
            )
            ax_pca_p.set_xlabel(f"PC1 ({var1_p*100:.2f}% variance)")
            ax_pca_p.set_ylabel(f"PC2 ({var2_p*100:.2f}% variance)")

            plt.tight_layout()
        else:
            print("avgpool embeddings not available — skipping PCA/regression summary.")

        # -----------------------------------------------------------------------
        # Spearman / rank correlation between PC1 & PC2 and the factor value
        # -----------------------------------------------------------------------
        # Use the avgpool embeddings (embeddings_p / param_values_p) because
        # they carry the factor values aligned with every image.

        rho_pc1_raw, p_pc1 = spearmanr(emb2_pca_p[:, 0], param_values_p)
        rho_pc2_raw, p_pc2 = spearmanr(emb2_pca_p[:, 1], param_values_p)
        # For this analysis, only correlation strength matters, not direction.
        rho_pc1 = abs(rho_pc1_raw)
        rho_pc2 = abs(rho_pc2_raw)

        print(f"\n  ── Spearman absolute correlation  [{model_path}]  factor={param_col} ──")
        print(f"     PC1 vs {param_col:25s}  |ρ| = {rho_pc1:.4f}   p = {p_pc1:.2e}")
        print(f"     PC2 vs {param_col:25s}  |ρ| = {rho_pc2:.4f}   p = {p_pc2:.2e}")

        # Annotate the param-colored PCA scatter with the correlation values
        ax_pca_p.annotate(
            f"Explained variance\n"
            f"PC1: {var1_p*100:.2f}%\n"
            f"PC2: {var2_p*100:.2f}%\n"
            f"Total: {(var1_p+var2_p)*100:.2f}%\n\n"
            f"|ρ|(PC1, {param_col}) = {rho_pc1:.3f}  (p={p_pc1:.1e})\n"
            f"|ρ|(PC2, {param_col}) = {rho_pc2:.3f}  (p={p_pc2:.1e})",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
        fig_pca_p.tight_layout()
        plt.close(fig_pca_p)

        # Store for summary heatmap
        if model_path not in spearman_results:
            spearman_results[model_path] = {}
        spearman_results[model_path][param_col] = {
            "PC1_rho": rho_pc1, "PC1_p": p_pc1,
            "PC2_rho": rho_pc2, "PC2_p": p_pc2,
            "EVR_PC1": var1_p, "EVR_PC2": var2_p,
            "EVR_TOTAL": (var1_p + var2_p),
        }

        # -----------------------------------------------------------------------
        # Linear regression analysis (separate from Spearman correlation)
        # -----------------------------------------------------------------------
        r2_pc1 = compute_linear_regression_r2(emb2_pca_p[:, [0]], param_values_p)
        r2_pc2 = compute_linear_regression_r2(emb2_pca_p[:, [1]], param_values_p)
        r2_pc12 = compute_linear_regression_r2(emb2_pca_p[:, :2], param_values_p)

        print(f"\n  ── Linear regression (R²)  [{model_path}]  factor={param_col} ──")
        print(f"     PC1 -> {param_col:25s}  R² = {r2_pc1:.4f}")
        print(f"     PC2 -> {param_col:25s}  R² = {r2_pc2:.4f}")
        print(f"     PC1+PC2 -> {param_col:21s}  R² = {r2_pc12:.4f}")

        if model_path not in linear_regression_results:
            linear_regression_results[model_path] = {}
        linear_regression_results[model_path][param_col] = {
            "PC1_r2": r2_pc1,
            "PC2_r2": r2_pc2,
            "PC12_r2": r2_pc12,
            "EVR_PC1": var1_p,
            "EVR_PC2": var2_p,
            "EVR_TOTAL": (var1_p + var2_p),
        }

        # parameter wise 2d
        # reducer = pacmap.PaCMAP(
        #     n_components=2,
        #     n_neighbors=10,
        #     MN_ratio=0.5,
        #     FP_ratio=2.0,
        #     random_state=42
        # )
        #
        # emb2 = reducer.fit_transform(embeddings)
        #
        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        # import numpy as np
        #
        # fig, ax = plt.subplots(figsize=(9, 8))
        #
        # sc = ax.scatter(
        #     emb2[:, 0],
        #     emb2[:, 1],
        #     c=param_values,
        #     cmap="viridis",
        #     s=8,
        # )
        #
        # cbar = fig.colorbar(sc, ax=ax)
        # cbar.set_label(param_col, fontsize=11)
        #
        # ax.set_title(f"2D PaCMAP — colored by {param_col}")
        # ax.set_xlabel("PaCMAP-1")
        # ax.set_ylabel("PaCMAP-2")
        #
        # plt.tight_layout()
        # plt.savefig(f"{model_path}/pacmap_parameter_wise_strawberry_{i}_2d.png", dpi=300)
        # plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Summary heatmap — Spearman |ρ| for PC1 and PC2 vs every factor
# One figure per model, rows = factors, columns = [PC1 ρ, PC2 ρ]
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np

for mdl_path, factor_dict in spearman_results.items():
    factors = list(factor_dict.keys())
    if not factors:
        continue

    pc1_rho = np.abs(np.array([factor_dict[f]["PC1_rho"] for f in factors]))
    pc2_rho = np.abs(np.array([factor_dict[f]["PC2_rho"] for f in factors]))
    pc1_p   = np.array([factor_dict[f]["PC1_p"]   for f in factors])
    pc2_p   = np.array([factor_dict[f]["PC2_p"]   for f in factors])
    evr_pc1 = np.array([factor_dict[f].get("EVR_PC1", np.nan) for f in factors])
    evr_pc2 = np.array([factor_dict[f].get("EVR_PC2", np.nan) for f in factors])
    evr_total = np.array([factor_dict[f].get("EVR_TOTAL", np.nan) for f in factors])

    # ── heatmap of ρ values ──────────────────────────────────────────────
    data = np.stack([pc1_rho, pc2_rho, evr_pc1, evr_pc2, evr_total], axis=1)   # (n_factors, 5)
    p_data = np.stack([pc1_p, pc2_p], axis=1)

    fig_h, ax_h = plt.subplots(figsize=(8.5, 0.6 * len(factors) + 1.8))
    im = ax_h.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")

    ax_h.set_xticks([0, 1, 2, 3, 4])
    ax_h.set_xticklabels(["PC1 |ρ|", "PC2 |ρ|", "EVR PC1", "EVR PC2", "EVR total"], fontsize=10, rotation=18, ha="right")
    ax_h.set_yticks(range(len(factors)))
    ax_h.set_yticklabels(factors, fontsize=9)

    # Annotate cells with ρ value; mark significant ones (p < 0.05) with *
    for row in range(len(factors)):
        for col in range(data.shape[1]):
            val  = data[row, col]
            star = ""
            if col < 2:
                pval = p_data[row, col]
                star = "*" if pval < 0.05 else ""
            ax_h.text(col, row, f"{val:.3f}{star}",
                      ha="center", va="center",
                      fontsize=8,
                      color="white" if val > 0.6 else "black")

    cbar_h = fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    cbar_h.set_label("|Spearman ρ| / Explained variance", fontsize=10)

    ax_h.set_title(
        f"Spearman absolute correlation + explained variance\n"
        f"Model: {mdl_path}  ({instance_suffix.replace('_', ' ')})\n(* p < 0.05)",
        fontsize=10,
    )
    fig_h.tight_layout()
    fig_h.subplots_adjust(bottom=0.22)

    heatmap_path = f"{mdl_path}/spearman_heatmap_banana_{instance_suffix}.png"
    fig_h.savefig(heatmap_path, dpi=150)
    plt.close(fig_h)
    print(f"Saved Spearman heatmap → {heatmap_path}")

    # ── bar chart: |ρ| PC1 vs PC2 side-by-side ───────────────────────────
    x   = np.arange(len(factors))
    w   = 0.35
    fig_b, ax_b = plt.subplots(figsize=(max(10, int(len(factors) * 1.2 + 3.0)), 5.2))
    bars1 = ax_b.bar(x - w / 2, pc1_rho, w, label="PC1 |ρ|",
                     color="#4C72B0", edgecolor="black", linewidth=0.6)
    bars2 = ax_b.bar(x + w / 2, pc2_rho, w, label="PC2 |ρ|",
                     color="#DD8452", edgecolor="black", linewidth=0.6)

    # Add explained variance (PC1+PC2) as a secondary-axis trend.
    ax_b2 = ax_b.twinx()
    evr_line = ax_b2.plot(x, evr_total, marker="o", linestyle="--", color="#55A868",
                          linewidth=1.6, label="EVR total")[0]

    # mark significant bars with a star above them
    for bar, pval in zip(list(bars1) + list(bars2),
                         list(pc1_p) + list(pc2_p)):
        if pval < 0.05:
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                "*", ha="center", va="bottom", fontsize=11, color="black",
            )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(factors, rotation=25, ha="right", fontsize=9)
    ax_b.set_ylim(0, 1.1)
    ax_b.set_ylabel("|Spearman ρ|", fontsize=11)
    ax_b2.set_ylim(0, 1.1)
    ax_b2.set_ylabel("Explained variance total", fontsize=11, color="#55A868")
    ax_b2.tick_params(axis="y", labelcolor="#55A868")
    ax_b.set_title(
        f"Embedding–factor alignment (|ρ|) + explained variance\n"
        f"Model: {mdl_path}  ({instance_suffix.replace('_', ' ')})  (* p < 0.05)",
        fontsize=11,
    )
    ax_b.legend([bars1, bars2, evr_line], ["PC1 |ρ|", "PC2 |ρ|", "EVR total"], fontsize=10)
    ax_b.grid(True, axis="y", alpha=0.3)
    fig_b.tight_layout()
    fig_b.subplots_adjust(bottom=0.25)

    bar_path = f"{mdl_path}/spearman_bar_banana_{instance_suffix}.png"
    fig_b.savefig(bar_path, dpi=150)
    plt.close(fig_b)
    print(f"Saved Spearman bar chart → {bar_path}")

    # ── print tidy table to console ───────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Spearman summary — model: {mdl_path}")
    print(f"{'─'*70}")
    print(f"  {'Factor':<28s}  {'PC1 |ρ|':>8s}  {'PC1 p':>10s}  {'PC2 |ρ|':>8s}  {'PC2 p':>10s}  {'EVR1':>7s}  {'EVR2':>7s}  {'EVRtot':>7s}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}")
    for f in factors:
        r = factor_dict[f]
        print(f"  {f:<28s}  {abs(r['PC1_rho']):>8.4f}  {r['PC1_p']:>10.2e}  "
              f"{abs(r['PC2_rho']):>8.4f}  {r['PC2_p']:>10.2e}  "
              f"{r.get('EVR_PC1', np.nan):>7.3f}  {r.get('EVR_PC2', np.nan):>7.3f}  {r.get('EVR_TOTAL', np.nan):>7.3f}")
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Summary plots — Linear regression R² for PC1/PC2/(PC1+PC2) vs every factor
# One figure per model, rows = factors
# ─────────────────────────────────────────────────────────────────────────────

for mdl_path, factor_dict in linear_regression_results.items():
    factors = list(factor_dict.keys())
    if not factors:
        continue

    pc1_r2 = np.array([factor_dict[f]["PC1_r2"] for f in factors])
    pc2_r2 = np.array([factor_dict[f]["PC2_r2"] for f in factors])
    pc12_r2 = np.array([factor_dict[f]["PC12_r2"] for f in factors])
    evr_pc1 = np.array([factor_dict[f].get("EVR_PC1", np.nan) for f in factors])
    evr_pc2 = np.array([factor_dict[f].get("EVR_PC2", np.nan) for f in factors])
    evr_total = np.array([factor_dict[f].get("EVR_TOTAL", np.nan) for f in factors])

    data = np.stack([pc1_r2, pc2_r2, pc12_r2, evr_pc1, evr_pc2, evr_total], axis=1)

    fig_lr_h, ax_lr_h = plt.subplots(figsize=(10.0, 0.6 * len(factors) + 1.8))
    im_lr = ax_lr_h.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

    ax_lr_h.set_xticks([0, 1, 2, 3, 4, 5])
    ax_lr_h.set_xticklabels(
        ["PC1 R²", "PC2 R²", "PC1+PC2 R²", "EVR PC1", "EVR PC2", "EVR total"],
        fontsize=10,
        rotation=18,
        ha="right",
    )
    ax_lr_h.set_yticks(range(len(factors)))
    ax_lr_h.set_yticklabels(factors, fontsize=9)

    for row in range(len(factors)):
        for col in range(data.shape[1]):
            val = data[row, col]
            ax_lr_h.text(
                col,
                row,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(val) > 0.55 else "black",
            )

    cbar_lr = fig_lr_h.colorbar(im_lr, ax=ax_lr_h, fraction=0.046, pad=0.04)
    cbar_lr.set_label("Linear regression R² / Explained variance", fontsize=10)

    ax_lr_h.set_title(
        f"Linear regression on PCA axes + explained variance\n"
        f"Model: {mdl_path}  ({instance_suffix.replace('_', ' ')})",
        fontsize=10,
    )
    fig_lr_h.tight_layout()
    fig_lr_h.subplots_adjust(bottom=0.24)

    lr_heatmap_path = f"{mdl_path}/linear_regression_heatmap_banana_{instance_suffix}.png"
    fig_lr_h.savefig(lr_heatmap_path, dpi=150)
    plt.close(fig_lr_h)
    print(f"Saved linear regression heatmap → {lr_heatmap_path}")

    x = np.arange(len(factors))
    w = 0.26
    fig_lr_b, ax_lr_b = plt.subplots(figsize=(max(10, int(len(factors) * 1.2 + 3.0)), 5.2))
    bars_lr1 = ax_lr_b.bar(x - w, pc1_r2, w, label="PC1 R²", color="#4C72B0", edgecolor="black", linewidth=0.6)
    bars_lr2 = ax_lr_b.bar(x, pc2_r2, w, label="PC2 R²", color="#DD8452", edgecolor="black", linewidth=0.6)
    bars_lr3 = ax_lr_b.bar(x + w, pc12_r2, w, label="PC1+PC2 R²", color="#8172B2", edgecolor="black", linewidth=0.6)

    ax_lr_b2 = ax_lr_b.twinx()
    evr_line_lr = ax_lr_b2.plot(
        x,
        evr_total,
        marker="o",
        linestyle="--",
        color="#55A868",
        linewidth=1.6,
        label="EVR total",
    )[0]

    ax_lr_b.set_xticks(x)
    ax_lr_b.set_xticklabels(factors, rotation=25, ha="right", fontsize=9)
    ax_lr_b.set_ylim(-1.0, 1.1)
    ax_lr_b.set_ylabel("Linear regression R²", fontsize=11)
    ax_lr_b2.set_ylim(0, 1.1)
    ax_lr_b2.set_ylabel("Explained variance total", fontsize=11, color="#55A868")
    ax_lr_b2.tick_params(axis="y", labelcolor="#55A868")
    ax_lr_b.set_title(
        f"Factor prediction from PCA axes (Linear regression R²)\n"
        f"Model: {mdl_path}  ({instance_suffix.replace('_', ' ')})",
        fontsize=11,
    )
    ax_lr_b.legend(
        [bars_lr1, bars_lr2, bars_lr3, evr_line_lr],
        ["PC1 R²", "PC2 R²", "PC1+PC2 R²", "EVR total"],
        fontsize=10,
    )
    ax_lr_b.grid(True, axis="y", alpha=0.3)
    fig_lr_b.tight_layout()
    fig_lr_b.subplots_adjust(bottom=0.25)

    lr_bar_path = f"{mdl_path}/linear_regression_bar_banana_{instance_suffix}.png"
    fig_lr_b.savefig(lr_bar_path, dpi=150)
    plt.close(fig_lr_b)
    print(f"Saved linear regression bar chart → {lr_bar_path}")

    print(f"\n{'═'*84}")
    print(f"  Linear regression summary — model: {mdl_path}")
    print(f"{'─'*84}")
    print(f"  {'Factor':<28s}  {'PC1 R²':>8s}  {'PC2 R²':>8s}  {'PC1+PC2 R²':>11s}  {'EVR1':>7s}  {'EVR2':>7s}  {'EVRtot':>7s}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*7}  {'─'*7}  {'─'*7}")
    for f in factors:
        r = factor_dict[f]
        print(
            f"  {f:<28s}  {r['PC1_r2']:>8.4f}  {r['PC2_r2']:>8.4f}  {r['PC12_r2']:>11.4f}  "
            f"{r.get('EVR_PC1', np.nan):>7.3f}  {r.get('EVR_PC2', np.nan):>7.3f}  {r.get('EVR_TOTAL', np.nan):>7.3f}"
        )
    print(f"{'═'*84}\n")


