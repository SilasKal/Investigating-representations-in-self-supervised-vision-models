import math
import os
from glob import glob
from collections import OrderedDict
from typing import Dict, List, Tuple, Callable, Optional, Union, Any

import torch
import torch.nn as nn
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.models import ResNet50_Weights
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import torchvision.transforms as transforms


# =========================
# 1) Model & preprocessing
# =========================

def get_model_and_preprocess(device: Optional[str] = None, model_name: str = "simclr"):
    """Load SimCLR ResNet and preprocessing pipeline; move to device."""
    from simclr import load_simclr
    if model_name == "simclr_santi":
        print("SimCLR Santi")
        model, preprocess = load_simclr()
        # print(model)
    elif model_name == "resnet50v1":
        print("ResNet50")
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "resnet50v2":
        print("ResNet50")
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
        # print(model)
        # preprocess = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(256),
        #     transforms.ToTensor(),
        # ])
    elif model_name == "mvimgnet_model":
        print("MVImgNet Model")
        from load_mvimgnet_model import load_mv_model
        model, preprocess = load_mv_model(path_weigths=r"C:\Users\silas\PycharmProjects\SimClr_MT\pretrained\weights\epoch_29.pt", device=device)
        # print(model)
    elif model_name in ["aasimclr", "simclrtt", "cipersimclr", "simclr"]:
        print(model_name)
        from load_aubret_models import mvimgnet
        model = mvimgnet(model_name)
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
        print(model)
    else:
        from load_aubret_models import custom_mvimgnet
        print("Custom MVImgNet Model")
        model = custom_mvimgnet(model_name)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    model.eval()
    if device is None:
        # device = 'cpu'
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, preprocess, device


# ===================
# 2) Hook utilities
# ===================

def register_activation_hooks(
        model: nn.Module, layer_names: List[str]
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    Register forward hooks on selected layers and return (activations_dict, handles).
    Stores CPU tensors in an OrderedDict keyed by layer name.
    """
    acts: Dict[str, torch.Tensor] = OrderedDict()
    modules = dict(model.named_modules())

    def make_hook(name: str) -> Callable:
        def hook(module, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            acts[name] = t.detach().cpu()

        return hook

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for name in layer_names:
        if name in modules:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        else:
            print(f"[warn] layer not found: {name}")
    return acts, handles


def register_relu_prepost_hooks(
        model: nn.Module, relu_names: List[str]
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    Register hooks on ReLUs to capture BOTH pre- and post-activation tensors.
    Returns (dict, handles) where dict has keys like 'layer4.2.relu/pre' and '/post'.
    """
    prepost: Dict[str, torch.Tensor] = OrderedDict()
    modules = dict(model.named_modules())
    def make_hook(name: str) -> Callable:
        def hook(module, inp, out):
            prepost[f"{name}/pre"] = inp[0].detach().cpu()
            prepost[f"{name}/post"] = out.detach().cpu()

        return hook

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for name in relu_names:
        if name in modules:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        else:
            print(f"[warn] relu not found: {name}")
    return prepost, handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove all hooks to avoid duplicates/memory leaks."""
    for h in handles:
        h.remove()


# ==================
# 3) Forward passes
# ==================

def forward_and_collect(model: nn.Module, x: torch.Tensor, acts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Run a forward pass; hooks fill `acts` in-place."""
    model.eval()
    with torch.no_grad():
        _ = model(x)
    for i, j in acts.items():
        print(f"  captured {i:20s} {tuple(j.shape)}")
    return acts


# ==========================
# 4) Introspection & stats
# ==========================

def summarize_activations(acts: Dict[str, torch.Tensor]) -> None:
    """Print shape, mean, std, and zero-ratio (sparsity) per watched layer."""
    for name, t in acts.items():
        mean = t.float().mean().item()
        std = t.float().std().item()
        sparsity = (t == 0).float().mean().item()
        print(f"{name:18s} shape={tuple(t.shape)}  mean={mean:.4f}  std={std:.4f}  sparsity={sparsity:.3f}")


def list_module_names(model: nn.Module, keyword_filters: Optional[List[str]] = None) -> List[str]:
    """Return module names (optionally filter by substrings)."""
    names = []
    for name, _ in model.named_modules():
        if keyword_filters is None or any(k in name for k in keyword_filters):
            names.append(name)
    return names


# ================
# 5) Visualization
# ================

def visualize_feature_maps(t: torch.Tensor, n: int = 8, title: str = "feature maps") -> None:
    """Visualize first n channels of [C,H,W] or [1,C,H,W]."""
    if t.dim() == 4:
        t = t[0]
    assert t.dim() == 3, f"Expected [C,H,W], got {tuple(t.shape)}"

    C, H, W = t.shape
    n = min(n, C)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        fm = t[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-6)
        axes[i].imshow(fm, interpolation="nearest")
        axes[i].axis("off")
    plt.suptitle(title)
    plt.show()


# ==========================
# 6) Targets & convenience
# ==========================

BLOCKS_PER_STAGE = {1: 3, 2: 4, 3: 6, 4: 3}


def build_block_targets(stage: int, submods: List[str]) -> List[str]:
    """All blocks in a stage for given submodules, e.g. ['conv3','bn3','relu']."""
    return [f"layer{stage}.{i}.{m}" for i in range(BLOCKS_PER_STAGE[stage]) for m in submods]


# ===========================================
# 7) ReLU → Global Average Pooled vector(s)
# ===========================================

def relu_vector_from_image(
        img_or_tensor: Union[Image.Image, torch.Tensor],
        relu_name: str = "layer4.2.relu",
        device: Optional[str] = None,
) -> torch.Tensor:
    """
    Returns a 1D vector [C] by taking the ReLU *post*-activation [B,C,H,W]
    and averaging over spatial dims (global average pooling).
    """
    model, preprocess, device = get_model_and_preprocess(device)

    # hook only the chosen ReLU and take POST (switch to '/pre' if needed)
    prepost, handles = register_relu_prepost_hooks(model, [relu_name])
    try:
        if isinstance(img_or_tensor, Image.Image):
            x = preprocess(img_or_tensor).unsqueeze(0).to(device)  # [1,3,H,W]
        elif isinstance(img_or_tensor, torch.Tensor):
            x = (img_or_tensor.unsqueeze(0) if img_or_tensor.dim() == 3 else img_or_tensor).to(device)
        else:
            raise TypeError("img_or_tensor must be PIL.Image or torch.Tensor")

        with torch.no_grad():
            _ = model(x)

        key = f"{relu_name}/post"  # or f"{relu_name}/pre" for pre-ReLU
        if key not in prepost:
            raise KeyError(f"Did not capture {key}. Check the relu_name.")
        act = prepost[key]  # [B,C,H,W] on CPU
        vec = act.mean(dim=[2, 3])[0]  # [C]
        return vec
    finally:
        remove_hooks(handles)


def vectors_from_folder(
        folder: str,
        relu_name: str = "layer4.2.relu",
        patterns: Tuple[str, ...] = ("*.jpg"),
        limit: Optional[int] = None,
        device: Optional[str] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Loads images from 'folder', computes a GAP vector per image from a chosen ReLU,
    returns (matrix [N,C] on CPU, list_of_paths).
    """
    paths = []
    for p in patterns:
        paths.extend(glob(os.path.join(folder, p)))
    paths.sort()
    print(f"Found {len(paths)} images in {folder} matching {patterns}")
    if limit is not None:
        paths = paths[:limit]

    vecs = []
    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
            v = relu_vector_from_image(img, relu_name=relu_name, device=device)  # [C]
            vecs.append(v.unsqueeze(0))  # [1,C]
            if i % 20 == 0:
                print(f"[{i}/{len(paths)}] processed")
        except Exception as e:
            print(f"[warn] failed {p}: {e}")

    if not vecs:
        raise RuntimeError("No vectors produced. Check folder and file types.")
    M = torch.cat(vecs, dim=0)  # [N,C]
    return M, paths


# ============================
# 8) Single-image demo runner
# ============================

def run_single_image_demo(
        img_or_tensor: Union[Image.Image, torch.Tensor],
        hook_mode: str = "stages",  # 'stages' | 'blocks' | 'prepost'
        include_downsample: bool = False,
        viz_layer: Optional[str] = "layer4",  # which layer to visualize (if present)
        device: Optional[str] = None,
        save_path: Optional[str] = None  # optional path to save a .pt dict of activations
) -> Dict[str, torch.Tensor]:
    """
    Full pipeline: load model, set hooks (by mode), forward, summarize, visualize.
    Returns a dict of activations (or pre/post for 'prepost' mode).
    """
    model, preprocess, device = get_model_and_preprocess(device)
    modules = dict(model.named_modules())
    print("Loaded model:\n", model)

    # ---- Build target lists
    if hook_mode == "stages":
        target_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
        acts, handles = register_activation_hooks(model, target_layers)

    elif hook_mode == "blocks":
        # capture conv3/bn3 across all blocks == near residual add point
        targets = []
        for s in (1, 2, 3, 4):
            targets += build_block_targets(s, ["conv3", "bn3"])
        if include_downsample:
            targets += [
                "layer2.0.downsample.0", "layer2.0.downsample.1",
                "layer3.0.downsample.0", "layer3.0.downsample.1",
                "layer4.0.downsample.0", "layer4.0.downsample.1",
            ]
        acts, handles = register_activation_hooks(model, targets)

    elif hook_mode == "prepost":
        # capture pre- and post-activation for final block of each stage + stem
        relu_targets = ["relu", "layer1.2.relu", "layer2.3.relu", "layer3.5.relu", "layer4.2.relu"]
        acts, handles = register_relu_prepost_hooks(model, relu_targets)
    else:
        raise ValueError("hook_mode must be in {'stages','blocks','prepost'}")

    # ---- Prepare input
    if isinstance(img_or_tensor, Image.Image):
        x = preprocess(img_or_tensor).unsqueeze(0).to(device)  # [1,3,256,256]
    elif isinstance(img_or_tensor, torch.Tensor):
        if img_or_tensor.dim() == 3:
            x = img_or_tensor.unsqueeze(0).to(device)
        elif img_or_tensor.dim() == 4:
            x = img_or_tensor.to(device)
        else:
            raise ValueError("Tensor must be [C,H,W] or [B,C,H,W]")
    else:
        raise TypeError("img_or_tensor must be a PIL.Image or torch.Tensor")

    try:
        # ---- Run
        forward_and_collect(model, x, acts)

        # ---- Summaries
        print("\n== Layer summaries ==")
        summarize_activations(acts)

        # ---- Visualization (only if not pre/post dict and target exists)
        if hook_mode != "prepost" and viz_layer is not None:
            candidate = None
            if viz_layer in acts:
                candidate = viz_layer
            else:
                for key in ("layer4.2.bn3", "layer4", "layer3.5.bn3"):
                    if key in acts:
                        candidate = key
                        break
            if candidate:
                visualize_feature_maps(acts[candidate], n=8, title=f"{candidate} feature maps")
            else:
                print(f"[info] No visualizable tensor found for viz_layer={viz_layer}")

        # ---- Optional save
        if save_path:
            torch.save(acts, save_path)
            print(f"[saved] activations to {save_path}")

        return acts
    finally:
        remove_hooks(handles)


def compute_rdm(M: torch.Tensor, metric: str = "euclidean", p: int = 1) -> torch.Tensor:
    """
    M: [N, C] matrix of per-image vectors (e.g., GAP on ReLU outputs).
    metric: 'euclidean'
    Returns: [N, N] dissimilarity matrix (larger = more dissimilar).
    """
    assert M.dim() == 2, "M must be [N, C]"
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    if metric == "euclidean":
        M_np = M.cpu().numpy()
        D = squareform(pdist(M_np, metric="euclidean"))
        return torch.from_numpy(D).to(M.device, dtype=M.dtype)
    else:
        raise ValueError("metric must be 'euclidean'")


def plot_rdm(rdm: torch.Tensor, title: str = "RDM"):
    """
    rdm: [N,N] torch tensor (CPU or CUDA). Plots as a heatmap.
    """
    if rdm.is_cuda:
        rdm = rdm.cpu()
    plt.figure(figsize=(5, 4))
    plt.imshow(rdm.numpy(), interpolation="nearest")
    plt.title(title)
    plt.xlabel("image index")
    plt.ylabel("image index")
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.show()


def compute_rdms_for_layers(
        vectors_by_layer: Dict[str, torch.Tensor],
        metric: str = "euclidean"
) -> Dict[str, torch.Tensor]:
    """
    vectors_by_layer: {layer_name: [N,C]}
    Returns: {layer_name: [N,N] RDM}
    """
    rdms = {}
    for ln, M in vectors_by_layer.items():
        rdms[ln] = compute_rdm(M, metric=metric)
    return rdms


from glob import glob
import csv


@torch.no_grad()
def layer_vectors_from_folder(
        folder: str,
        render_param: str,
        param_index: int = 0,
        layer_names: List[str] = ("layer1", "layer2", "layer3", "layer4"),
        patterns: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png"),
        limit: Optional[int] = None,
        device: Optional[str] = None,
        model_name: str = "simclr"
) -> tuple[dict[str, Tensor], list[Any], list[Any]]:
    """
    For each image in `folder`, capture the selected layer outputs, do global average pooling
    over spatial dims (H,W) to get a vector [C] per layer. Returns:
        vectors: {layer_name: [N, C] tensor on CPU}
        paths:   list of image paths in order
    Notes:
    - These layer_names should be modules that output [B,C,H,W] and are *after* ReLUs
      (e.g., 'layer1'...'layer4' as in your ResNet).
    """
    model, preprocess, device = get_model_and_preprocess(device, model_name)

    # Register hooks ONCE for all target layers
    acts, handles = register_activation_hooks(model, list(layer_names))
    try:
        order = []
        with open(render_param, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Überspringe Kopfzeile
            for row in reader:
                order.append(row[param_index])
        # Sammle Bildpfade
        # print(order)
        paths = []
        for p in patterns:
            paths.extend(glob(os.path.join(folder, p)))
        paths = [x for _, x in sorted(zip(order, paths), key=lambda pair: pair[0], reverse=True)]
        if limit is not None:
            paths = paths[:limit]
            order = order[:limit]
        order = [float(x) for x in order]
        # print(paths)

        if not paths:
            raise RuntimeError(f"No images found in {folder} matching {patterns} and sort_order.csv")

        # Per-layer buffers
        buf: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
        print(device)
        for i, p in enumerate(paths, 1):
            try:
                img = Image.open(p).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W] on model device
                _ = model(x)  # fills `acts` with {layer_name: [1,C,H,W]}
                # vectorize each hooked layer (GAP over H,W)
                for ln in layer_names:
                    if ln not in acts:
                        raise KeyError(f"Missing activation for {ln}. Check layer name.")
                    t = acts[ln]  # [1,C,H,W] on CPU (hooks detach to CPU)
                    v = t.mean(dim=[2, 3])[0]  # [C]
                    buf[ln].append(v.unsqueeze(0))  # keep batch dim for later cat
            except Exception as e:
                print(f"[warn] failed {p}: {e}")
            finally:
                acts.clear()  # clear for the next image to keep memory small

            if i % 100 == 0:
                print(f"[{i}/{len(paths)}] processed")
    finally:
        remove_hooks(handles)
    # Concatenate per-layer buffers
    vectors: Dict[str, torch.Tensor] = {}
    for ln in layer_names:
        vectors[ln] = torch.cat(buf[ln], dim=0)  # [N,C]
    return vectors, paths, order


import torch


def rdm_uppervec(rdm: torch.Tensor) -> torch.Tensor:
    """
    Convert a square [N,N] RDM to a 1D vector containing the upper-triangular
    entries (excluding the diagonal). Output length = N*(N-1)/2.
    """
    if rdm.dim() != 2 or rdm.size(0) != rdm.size(1):
        raise ValueError("rdm must be square [N,N]")
    N = rdm.size(0)
    iu = torch.triu_indices(N, N, offset=1)
    v = rdm[iu[0], iu[1]]
    return v.reshape(-1)


from scipy.stats import pearsonr


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Pearson-Korrelation zwischen zwei 1D-Tensoren (mit scipy).
    """
    if a.numel() != b.numel():
        raise ValueError("vectors must have same length")
    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    corr, _ = pearsonr(a_np, b_np)
    return float(corr)


def rsa_matrix_from_rdms(
        rdms: dict[str, torch.Tensor],
        method: str = "pearson"
) -> torch.Tensor:
    """
    rdms: {layer_name: [N,N] RDM} (same N across layers/images!)
    method: 'pearson'
    Returns:
        R: [L,L] tensor, where L = number of layers (order matches sorted keys)
    """
    if method.lower() not in {"pearson"}:
        raise ValueError("method must be 'pearson'")
    names = sorted(rdms.keys())
    vecs = []
    for n in names:
        v = rdm_uppervec(rdms[n]).to(torch.float32)
        # optional: z-score each vector to equalize scale (Pearson already mean/scales)
        vecs.append(v)
    L = len(names)
    R = torch.empty(L, L, dtype=torch.float32)
    corr_fn = _pearson_corr
    for i in range(L):
        R[i, i] = 1.0
        for j in range(i + 1, L):
            c = corr_fn(vecs[i], vecs[j])
            R[i, j] = R[j, i] = c
    return R, names


def plot_rsa(R: torch.Tensor, names: list[str], title: str = "RSA (layer × layer)"):
    """
    Heatmap for RSA matrix. R: [L,L], names: list of layer names in order.
    """
    import matplotlib.pyplot as plt
    if R.is_cuda:
        R = R.cpu()
    plt.figure(figsize=(6, 5))
    plt.imshow(R.numpy(), vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(shrink=0.8, label="correlation")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


def reduce_layers(
        vectors_by_layer: dict[str, torch.Tensor],
        method: str = "pca",  # 'pca' | 'isomap' | 'pca_then_isomap'
        n_components: int = 3,  # >=3 for 'pca'/'isomap'; ignored for 'pca_then_isomap' (fixed to 2D)
        max_points: int = 20000,
        random_state: int = 42,
        isomap_n_neighbors: int = 20,
        return_embeddings: bool = False,
        pca_then_isomap_components: int = 10,  # PCA stage components for the pipeline
):
    """
    Runs dimensionality reduction per layer without plotting.

    Modes:
      - 'pca'              : PCA to n_components (>=3). Prints explained variance ratios.
      - 'isomap'           : Isomap to n_components (>=3). Prints completion info.
      - 'pca_then_isomap'  : PCA -> 20 dims (config via pca_then_isomap_components), then Isomap -> 2 dims.
                             Prints PCA stage explained variance (sum + first few ratios).

    Returns:
      dict {layer: embedding ndarray} if return_embeddings=True, else None.
    """
    rng = torch.Generator().manual_seed(random_state)
    out_Z = {} if return_embeddings else None

    for layer, feats in vectors_by_layer.items():
        X = feats.cpu()
        N = len(X)

        # Optional subsample for speed
        if N > max_points:
            idx = torch.randperm(N, generator=rng)[:max_points]
            X = X[idx]

        Xn = X.numpy()
        mode = method.lower()

        if mode == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            Z = reducer.fit_transform(Xn)
            var = reducer.explained_variance_ratio_
            print(f"[PCA]  {layer}: components={n_components}  "
                  f"var_ratio(first 5)={var[:5].tolist()}  sum={var.sum():.4f}")

        elif mode == "isomap":
            reducer = Isomap(n_neighbors=isomap_n_neighbors, n_components=n_components)
            Z = reducer.fit_transform(Xn)
            print(f"[ISOMAP] {layer}: components={n_components}, "
                  f"n_neighbors={isomap_n_neighbors} (explained variance not defined)")

        elif mode == "pca_then_isomap":
            # Stage 1: PCA -> k dims (ensure k >= target isomap dims)
            k = int(pca_then_isomap_components)
            n_iso = int(n_components)
            if k < 2:
                raise ValueError("pca_then_isomap_components must be >= 2")
            if k < n_iso:
                # ensure PCA does not reduce below the desired Isomap output dim
                k = n_iso
                print(f"[PCA→ISOMAP] {layer}: bumped PCA k to {k} to match Isomap target {n_iso}D")
            pca_stage = PCA(n_components=k, random_state=random_state)
            Zp = pca_stage.fit_transform(Xn)
            var = pca_stage.explained_variance_ratio_
            print(f"[PCA→ISOMAP] {layer}: PCA k={k}  var_ratio(first 5)={var[:5].tolist()}  sum={var.sum():.4f}")

            # Stage 2: Isomap -> n_iso D (robust neighbor clamping)
            if Zp.shape[0] < 2:
                raise ValueError(f"Not enough samples ({Zp.shape[0]}) for Isomap")
            # n_neighbors must be <= n_samples - 1 and at least 1
            max_allowed_neighbors = max(1, Zp.shape[0] - 1)
            k_neighbors = min(isomap_n_neighbors, max_allowed_neighbors)
            iso_stage = Isomap(n_neighbors=k_neighbors, n_components=n_iso)
            Z = iso_stage.fit_transform(Zp)
            print(f"[PCA→ISOMAP] {layer}: Isomap to {n_iso}D (n_neighbors={k_neighbors}) complete")


        # if n_components == 2:
        #     # plot 2D scatter
        #     # plt.figure(figsize=(5, 4))
        #     # plt.scatter(Z[:, 0], Z[:, 1], s=10)
        #     # plt.title(f"{method.upper()} 2D projection of {layer} ({len(X)} points)")
        #     # plt.xlabel("Component 1")
        #     # plt.ylabel("Component 2")
        #     # plt.tight_layout()
        #     # plt.show()
        #     # plot_embedding_cyclic(Z, title = layer)
        #     print("2D plot skipped in this context.")
        #     pass
        else:
            raise ValueError("method must be 'pca', 'isomap', or 'pca_then_isomap'")

        if return_embeddings:
            out_Z[layer] = Z

    return out_Z


import numpy as np
import matplotlib.pyplot as plt


def plot_embedding_cyclic(emb2d: np.ndarray, title: str = "Hue"):
    """
    emb2d: ndarray [N,2]. Colors points by angle atan2(y,x), using a cyclic colormap.
    """
    X = emb2d
    x, y = X[:, 0], X[:, 1]
    ang = np.arctan2(y, x)  # [-pi, pi]
    ang01 = (ang % (2 * np.pi)) / (2 * np.pi)  # [0,1] wrap-around from 0 to 2pi
    plt.figure(figsize=(5.5, 5))
    sc = plt.scatter(x, y, c=ang01, s=10, cmap="hsv")  # 'hsv' for hue, 'twilight' camera
    cbar = plt.colorbar(sc, shrink=0.5, ticks=[0, 0.5, 1.0])
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])
    plt.title(title)
    plt.tight_layout()
    plt.show()


import torch
from typing import Dict, Tuple, List
from scipy.stats import pearsonr


def _pearson_corr_vecs(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np, b_np = a.cpu().numpy(), b.cpu().numpy()
    r, _ = pearsonr(a_np, b_np)
    return float(r)


def cross_rsa_from_rdms(
        rdms_A: Dict[str, torch.Tensor],
        rdms_B: Dict[str, torch.Tensor],
        method: str = "pearson",
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    rdms_A: {layer_name_A: [N,N]} for condition A (e.g., color)
    rdms_B: {layer_name_B: [N,N]} for condition B (e.g., camera)
    Returns:
      R: [L_A, L_B] correlation matrix between upper-tri vectors of each pair (A_i vs B_j)
      names_A, names_B: layer orders used
    """
    assert method.lower() == "pearson", "Only 'pearson' implemented here."
    names_A = sorted(rdms_A.keys())
    names_B = sorted(rdms_B.keys())

    vecs_A = [rdm_uppervec(rdms_A[n]).to(torch.float32) for n in names_A]
    vecs_B = [rdm_uppervec(rdms_B[n]).to(torch.float32) for n in names_B]

    # sanity: same number of pairwise entries
    L = len(vecs_A[0])
    assert all(v.numel() == L for v in vecs_A), "RDMs in A have mismatched sizes."
    assert all(v.numel() == L for v in vecs_B), "RDMs in B have mismatched sizes."

    R = torch.empty(len(names_A), len(names_B), dtype=torch.float32)
    for i, va in enumerate(vecs_A):
        for j, vb in enumerate(vecs_B):
            R[i, j] = _pearson_corr_vecs(va, vb)
    return R, names_A, names_B


def plot_cross_rsa(R: torch.Tensor, names_A: List[str], names_B: List[str],
                   title: str = "Cross-condition RSA (A vs B)"):
    import matplotlib.pyplot as plt
    if R.is_cuda: R = R.cpu()
    plt.figure(figsize=(6.5, 5.2))
    plt.imshow(R.numpy(), vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(shrink=0.8, label="correlation")
    plt.xticks(range(len(names_B)), names_B, rotation=45, ha="right")
    plt.yticks(range(len(names_A)), names_A)
    plt.title(title)
    plt.tight_layout()
    plt.show()




# =================
# run
# =================

if __name__ == "__main__":
    # Folder with images
    # model_name = "cipersimclr"
    # images_folder = r"C:\Users\silas\PycharmProjects\SimClr_MT\umbrella\camera\images"
    # #
    # # # Choose layers to summarize (these are the post-ReLU stage outputs in your ResNet)
    # # layers = ["layer1.0.relu", "layer1.1.relu", "layer1.2.relu", "layer2.0.relu", "layer2.1.relu", "layer2.2.relu", "layer2.3.relu", "layer3.0.relu", "layer3.1", "layer3.2.relu", "layer3.3.relu", "layer3.4.relu", "layer3.5.relu", "layer4.0.relu", "layer4.1.relu", "layer4.2.relu"]
    # # # layers = ["layer1","layer2","layer3", "layer4", "conv1", "bn1", "relu", "maxpool"]
    # # # layers = ["layer4","layer3","layer2", "layer1"]
    # # # layers = ["layer1.0.relu", "layer1.1.relu", "layer1.2.relu", "layer2.0.relu", "layer2.1.relu", "layer2.2.relu", "layer2.3.relu", "layer3.0.relu", "layer3.1.relu", "layer3.2.relu", "layer3.3.relu", "layer3.4.relu", "layer3.5.relu", "layer4.0.relu", "layer4.1.relu", "layer4.2.relu"]
    # layers = ["relu", "layer1.0.relu", "layer2.0.relu", "layer3.0.relu", "layer4.0.relu", "layer4.1.relu", "layer4.2.relu"]
    # # # 1) Build per-image vectors per layer (ReLU outputs → GAP)
    # vectors_by_layer, paths, camera_angles = layer_vectors_from_folder(
    #     images_folder,
    #     render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\umbrella\camera\render_params.csv",
    #     layer_names=layers,
    #     limit=None,  # or None for all
    #     model_name = model_name
    # )
    # # for ln, M in vectors_by_layer.items():
    # #     print(f"{ln}: vectors matrix {tuple(M.shape)}")  # e.g., [N, C]
    # #
    # # # 2) Compute RDMs (choose: 'euclidean', 'cosine', or 'correlation')
    # # rdms = compute_rdms_for_layers(vectors_by_layer, metric="euclidean")
    # # print(rdms.keys())
    # # # # 3) Plot a couple of them
    # # # for i in layers:
    # # #     plot_rdm(rdms[i], title="RDM" + i)
    # # #
    # # #
    # # # R, layer_order = rsa_matrix_from_rdms(rdms, method="pearson")  # or "pearson"
    # # # plot_rsa(R, layer_order, title="RSA across layers (Pearson, Euclidean RDMs)")
    # #
    # # visualize each layer with PCA (2D)
    # embedding = reduce_layers(vectors_by_layer, method="pca_then_isomap", n_components=2, return_embeddings=True)
    # print(embedding.keys())
    # # colorbar range based on min/max of camera_angles/hue_values
    # # vmin = float(np.min(hue_values))
    # # vmax = float(np.max(hue_values))
    # vmin = float(np.min(camera_angles))
    # vmax = float(np.max(camera_angles))
    # # print(alphas)
    # for idx, (ln, Z) in enumerate(embedding.items()):
    #     sc = plt.scatter(Z[:, 0], Z[:, 1], s=10, c=camera_angles, cmap="twilight", label=ln, vmin=vmin,
    #                     vmax=vmax)
    #     # sc = plt.scatter(Z[:, 0], Z[:, 1], s=10, c=hue_values, cmap="hsv", label=ln, vmin=vmin,
    #     #                 vmax=vmax)
    #     plt.xlabel("Component 1")
    #     plt.ylabel("Component 2")
    #     cbar = plt.colorbar(sc, shrink=0.5, ticks=[0, math.pi, 2 * math.pi])
    #     cbar.set_ticks([0, math.pi, 2 * math.pi])
    #     cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])
    #     # plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"umbrella_camera_{model_name}_{str(layers[idx])}_pca_isomap.png", dpi=300)
    #     plt.show()

    # cross RSM with camera and hue
    # layers = ["layer1.0.relu", "layer1.1.relu", "layer1.2.relu", "layer2.0.relu", "layer2.1.relu", "layer2.2.relu", "layer2.3.relu", "layer3.0.relu", "layer3.1.relu", "layer3.2.relu", "layer3.3.relu", "layer3.4.relu", "layer3.5.relu", "layer4.0.relu", "layer4.1.relu", "layer4.2.relu"]
    # # # 1) Build per-image vectors per layer (ReLU outputs → GAP)
    # #
    # model_name = "cipersimclr"
    # layers = ["layer1.0.relu", "layer2.0.relu", "layer3.0.relu", "layer4.0.relu", "layer4.1.relu", "layer4.2.relu"]
    # vectors_by_layer, paths, _ = layer_vectors_from_folder(
    #     folder=r"C:\Users\silas\PycharmProjects\SimClr_MT\umbrella\hue\images",
    #     render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\umbrella\hue\render_params.csv",
    #     layer_names=layers,
    #     limit=None,  # or None for all
    #     model_name = model_name
    # )
    # vectors_by_layer2, paths2, _ = layer_vectors_from_folder(
    #     r"C:\Users\silas\PycharmProjects\SimClr_MT\umbrella\camera\images",
    #     render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\umbrella\camera\render_params.csv",
    #     layer_names=layers,
    #     limit=None,  # or None for all
    #     model_name = model_name
    # )
    # # # rename layers to distinguish
    # vectors_by_layer = {f"{k}_hue": v for k, v in vectors_by_layer.items()}
    # vectors_by_layer2 = {f"{k}_cam": v for k, v in vectors_by_layer2.items()}
    # rdms = compute_rdms_for_layers(vectors_by_layer, metric="euclidean")
    # rdms2 = compute_rdms_for_layers(vectors_by_layer2, metric="euclidean")
    # R, names_A, names_B = cross_rsa_from_rdms(rdms, rdms2, method="pearson")
    # # plot_cross_rsa(R, names_A, names_B, title="Cross RSA Hue vs Camera (Pearson, Euclidean RDMs)")
    # #
    # # # persistence diagram
    # from ripser import ripser
    # from persim import plot_diagrams, bottleneck
    #
    # emb_hue_dict = reduce_layers(vectors_by_layer, method="pca_then_isomap", n_components=2, return_embeddings=True)
    # emb_cam_dict = reduce_layers(vectors_by_layer2, method="pca_then_isomap", n_components=2, return_embeddings=True)
    #
    # for key in emb_hue_dict.keys():
    #     if not key.endswith("_hue"):
    #         continue
    #     base = key[:-4]  # entferne suffix "_hue"
    #     cam_key = f"{base}_cam"
    #     if cam_key not in emb_cam_dict:
    #         print(f"[warn] missing {cam_key} in emb_cam_dict")
    #         continue
    #
    #     emb_hue = emb_hue_dict[key]  # ndarray [N,2]
    #     emb_cam = emb_cam_dict[cam_key]  # ndarray [N,2]
    #
    #     # # (Optional) center before topology to stabilize angle-based rings
    #     # emb_hue = emb_hue - emb_hue.mean(axis=0, keepdims=True)
    #     # emb_cam = emb_cam - emb_cam.mean(axis=0, keepdims=True)
    #
    #     # ---- 3) Persistent homology on 2D embeddings ----
    #     res_hue = ripser(emb_hue, maxdim=1)
    #     res_cam = ripser(emb_cam, maxdim=1)
    #
    #     # Plot Hue diagram
    #     plot = plt.figure()
    #     plot_diagrams(res_hue["dgms"])
    #     plt.title(f"Persistence Diagram Hue - {base}")
    #     plt.tight_layout()
    #     plt.savefig(f'persistence_diagram_hue_{model_name}_{base}.png', dpi=300)
    #     plot.show()
    #
    #     plot = plt.figure()
    #     plot_diagrams(res_cam["dgms"])
    #     plt.title(f"Persistence Diagram Camera - {base}")
    #     plt.tight_layout()
    #     plt.savefig(f'persistence_diagram_camera_{model_name}_{base}.png', dpi=300)
    #     plot.show()

    # camera x hue
    import matplotlib
    # layers = ["relu"]
    # layers = ["layer1.0.relu"]
    # layers = ["layer2.0.relu"]
    # layers = ["layer3.0.relu"]
    # layers = ["layer4.0.relu"]
    # layers = ["layer4.1.relu"]
    # layers = ["layer4.2.relu"]
    # model_name = "aasimclr"
    model_name = r"C:\Users\silas\PycharmProjects\SimClr_MT\models_finetuning_MAPS_10k\epoch_49.pt"
    layers = ["relu", "layer1.0.relu", "layer2.0.relu","layer3.0.relu", "layer4.0.relu", "layer4.1.relu", "layer4.2.relu"]
    # images_folder = r"C:\Users\silas\PycharmProjects\SimClr_MT\camera_hue\images"
    images_folder = r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\001_goldfish\images"
    param_index = 4
    render_param = r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\001_goldfish\parameters.csv"
    vectors_by_layer_ordered_by_camera, paths, camera_angles = layer_vectors_from_folder(
        images_folder,
        param_index=param_index,
        # render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\umbrella\camera\render_params.csv",
        # render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\camera_hue\render_params.csv",
        render_param=render_param,
        layer_names=layers,
        limit=20000,
        model_name=model_name
    )
    # # matplotlib.use('qtagg')
    # # vectors_by_layer_ordered_by_hue, paths2, hue_values = layer_vectors_from_folder(
    # #     images_folder,
    # #     param_index=1,
    # #     render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\camera_hue\render_params.csv",
    # #     layer_names=layers,
    # #     limit=20000,
    # #     model_name=model_name
    # # )
    embedding_ord_camera = reduce_layers(
        vectors_by_layer_ordered_by_camera,
        method="pca_then_isomap",
        n_components=3,
        return_embeddings=True
    )
    #
    # # embedding_ord_hue = reduce_layers(
    # #     vectors_by_layer_ordered_by_hue,
    # #     method="pca_then_isomap",
    # #     n_components=3,
    # #     return_embeddings=True,
    # # )
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = None
    #
    # colorbar range based on min/max of camera_angles/hue_values
    # # vmin = float(np.min(hue_values))
    # # vmax = float(np.max(hue_values))
    vmin = float(np.min(camera_angles))
    vmax = float(np.max(camera_angles))
    for idx, (ln, Z) in enumerate(embedding_ord_camera.items()):
        N = Z.shape[0]
        sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=10, c=camera_angles, cmap="twilight", label=ln, vmin=vmin, vmax=vmax)
        # sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=10, c=hue_values, cmap="hsv", label=ln, vmin=vmin,
        #                 vmax=vmax)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    if sc is not None:
        ticks = [vmin, (vmin + vmax) / 2.0, vmax]
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, ticks=ticks)
        cbar.set_ticklabels([f"{t:.3f}" for t in ticks])
    # plt.legend()
    plt.tight_layout()
    # plt.savefig(f'camera_x_hue_3d_scatter_camera_{model_name}_{str(layers)}.png', dpi=300)
    plt.savefig(f'camera_x_hue_3d_scatter_goldfish_{param_index}_10k_finetuned_{str(layers[idx])}.png', dpi=300)
    plt.show()

    # persistence diagram camera x hue (3D)
    from ripser import ripser
    from persim import plot_diagrams, bottleneck
    #
    # number of points to subsample for ripser
    SAMPLE_SIZE = 1000
    # model_name = "cipersimclr"
    layers = ["relu", "layer1.0.relu","layer2.0.relu", "layer3.0.relu", "layer4.0.relu","layer4.1.relu", "layer4.2.relu"]
    # images_folder = r"C:\Users\silas\PycharmProjects\SimClr_MT\650_microphone\650_microphone\images"
    # images_folder = r"C:\Users\silas\PycharmProjects\SimClr_MT\camera_hue\images"
    vectors_by_layer_ordered_by_camera, paths, camera_angles = layer_vectors_from_folder(
        images_folder,
        param_index=param_index,
        # render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\650_microphone\650_microphone\params.csv",
        # render_param=r"C:\Users\silas\PycharmProjects\SimClr_MT\camera_hue\render_params.csv",
        render_param=render_param,
        layer_names=layers,
        limit=None,
        model_name=model_name
    )
    embedding_ord_camera = reduce_layers(
        vectors_by_layer_ordered_by_camera,
        method="pca_then_isomap",
        n_components=3,
        return_embeddings=True
    )

    # colorbar range based on min/max of camera_angles/hue_values
    # vmin = float(np.min(hue_values))
    # vmax = float(np.max(hue_values))
    vmin = float(np.min(camera_angles))
    vmax = float(np.max(camera_angles))
    for idx, (ln, Z) in enumerate(embedding_ord_camera.items()):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = None
        N = Z.shape[0]
        sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=10, c=camera_angles, cmap="twilight", label=ln, vmin=vmin, vmax=vmax)
        # sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=10, c=hue_values, cmap="hsv", label=ln, vmin=vmin,
        #                 vmax=vmax)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

        if sc is not None:
            ticks = [vmin, (vmin + vmax) / 2.0, vmax]
            cbar = fig.colorbar(sc, ax=ax, shrink=0.5, ticks=ticks)
            cbar.set_ticklabels([f"{t:.3f}" for t in ticks])
        # plt.legend()
        plt.tight_layout()
        # plt.savefig(f'camera_x_hue_3d_scatter_camera_microphone_{model_name}_{str(layers[idx])}.png', dpi=300)
        plt.savefig(f'camera_x_hue_3d_scatter_camera_goldfish_{param_index}_10k_finetuned_{str(layers[idx])}.png', dpi=300)
        plt.show()
    for key in embedding_ord_camera.keys():
        base = key
        emb_cam = embedding_ord_camera[base]  # ndarray [N,3]
        N = emb_cam.shape[0]

        # -uniform subsample to SAMPLE_SIZE points for ripser (if needed) -
        if N > SAMPLE_SIZE:
            idx = np.linspace(0, N - 1, num=SAMPLE_SIZE, dtype=int)
            emb_sample = emb_cam[idx]
            print(f"[info] {base}: subsampled {N} → {len(idx)} points for ripser")
        else:
            emb_sample = emb_cam
            print(f"[info] {base}: using all {N} points for ripser")

        # ---- 3) Persistent homology on 3D embeddings (subsampled) ----
        res_cam = ripser(emb_sample, maxdim=2)

        # Plot diagrams
        plot2 = plt.figure()
        plot_diagrams(res_cam["dgms"])
        plt.title(f"{base} Camera topology (3D embedding, N={emb_sample.shape[0]})")
        plt.tight_layout()
        plot2.show()
        # plot2.savefig(f"persistence_diagram_camera_3d_microphone_{model_name}_{base}.png", dpi=300)
        plot2.savefig(f"persistence_diagram_camera_3d_{param_index}_goldfish_10k_finetuned_{base}.png", dpi=300)


### classification heat map

import os, re
from glob import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import Tuple, Optional

_IDX_RE = re.compile(r"(\d+)(?=\.(jpg|jpeg|png)$)", re.IGNORECASE)

def _idx_from_name(p: str) -> int:
    m = _IDX_RE.search(os.path.basename(p))
    return int(m.group(1)) if m else 10**18

@torch.no_grad()
def outputs_from_folder(
    folder: str,
    render_param: str,
    param_indices: Tuple[int, int] = (0, 1),   # (camera_idx, hue_idx)
    patterns: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png"),
    limit: Optional[int] = None,
    device: Optional[str] = None,
    model_name: str = "resnet50",
):
    model, preprocess, device = get_model_and_preprocess(device, model_name)

    # images sorted by index
    paths = []
    for p in patterns:
        paths.extend(glob(os.path.join(folder, p)))
    paths.sort(key=_idx_from_name)

    # csv in row order
    df = pd.read_csv(render_param)
    N = min(len(paths), len(df))
    if limit is not None:
        N = min(N, int(limit))
    paths = paths[:N]
    df = df.iloc[:N].reset_index(drop=True)

    p0 = df.iloc[:, param_indices[0]].to_numpy(dtype=float)
    p1 = df.iloc[:, param_indices[1]].to_numpy(dtype=float)

    preds = np.empty(N, dtype=int)
    preds_unchanged = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        y = model(x)
        # print(y.shape)
        if isinstance(y, (tuple, list)): y = y[0]
        preds[i] = int(torch.argmax(y, dim=1).item())
        preds_unchanged.append(y.detach())
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"[info] processed {i + 1}/{N} images")

    return preds, p0, p1, paths, preds_unchanged

import numpy as np
import matplotlib.pyplot as plt

def heatmap_from_params(
    preds: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    target_class_id: int,
    bins0: int = 180,     # z.B. camera steps
    bins1: int = 180,     # z.B. hue steps
    p0_range=None,        # (min,max) optional
    p1_range=None,
):
    preds = np.asarray(preds)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    acc = (preds == int(target_class_id)).astype(np.float32)

    if p0_range is None: p0_range = (float(p0.min()), float(p0.max()))
    if p1_range is None: p1_range = (float(p1.min()), float(p1.max()))

    # bin edges
    e0 = np.linspace(p0_range[0], p0_range[1], bins0 + 1)
    e1 = np.linspace(p1_range[0], p1_range[1], bins1 + 1)

    # accumulate sum + count per cell
    H_sum = np.zeros((bins0, bins1), dtype=np.float32)
    H_cnt = np.zeros((bins0, bins1), dtype=np.int32)

    i0 = np.clip(np.digitize(p0, e0) - 1, 0, bins0 - 1)
    i1 = np.clip(np.digitize(p1, e1) - 1, 0, bins1 - 1)

    for a, r, c in zip(acc, i0, i1):
        H_sum[r, c] += a
        H_cnt[r, c] += 1

    H_acc = np.divide(H_sum, np.maximum(H_cnt, 1), dtype=np.float32)  # 0..1
    return H_acc, e0, e1

def plot_heatmap(H_acc, e0, e1, title="Camera × Hue", xlabel="p1", ylabel="p0", out_path=None):
    plt.figure(figsize=(6,5))
    # imshow expects [rows, cols] -> rows correspond to p0 bins, cols to p1 bins
    extent = [e1[0], e1[-1], e0[0], e0[-1]]  # x=p1, y=p0
    im = plt.imshow(H_acc, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    plt.colorbar(im, label="Accuracy")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        plt.show()

def santi_plot(target_class, preds, camera, hue, outpath):
    from scipy.special import softmax
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    num_label_camera = len(np.unique(camera))
    num_label_hue = len(np.unique(hue))
    preds = np.array([p.detach().cpu().numpy() for p in preds])
    accuracy = softmax(preds, axis=1) # error!!!
    accuracy = accuracy.squeeze()
    # print how many values are not 1
    print(f"{np.sum(accuracy != 1)=}")
    print(f"{np.unique(np.argmax(accuracy, axis=1))=}")
    print(accuracy.shape)
    accuracy_umbrella = accuracy[:, target_class]
    print(accuracy_umbrella.shape)
    print(accuracy_umbrella)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(accuracy_umbrella.reshape(num_label_camera, num_label_hue), vmin=0, vmax=1, cmap='plasma')
    ax.invert_yaxis()
    ax.set_yticks([0, 49, 99], [0, r'$\pi$', r'2$\pi$'])
    ax.set_xticks([0, 49, 99], [0, r'$\pi$', r'2$\pi$'])
    ax.set_xlabel(r'$\theta_{hue}$', fontsize=12)
    ax.set_ylabel(r'$\theta_{camera}$', fontsize=12)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Accuracy', rotation=270, labelpad=15)
    fig.savefig(outpath)
    plt.show()
    idx_min = np.argmin(accuracy_umbrella)
    camera_min = camera[idx_min]
    hue_min = hue[idx_min]

    print(camera_min, hue_min)



# classification heatmap run
# target_classes = [879, 650]
# folder_paths = [r"C:\Users\silas\PycharmProjects\SimClr_MT\camera_hue\images",
#                 r"C:\Users\silas\PycharmProjects\SimClr_MT\650_microphone\650_microphone\images"]
# render_params= [r"C:\Users\silas\PycharmProjects\SimClr_MT\camera_hue\render_params.csv",
#                 r"C:\Users\silas\PycharmProjects\SimClr_MT\650_microphone\650_microphone\params.csv"]
# bin_sizes = [48, 24]
target_classes = [1, 293, 404, 541, 579, 651, 671, 732, 949, 954]
folder_paths = [r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\001_goldfish\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\293_cheetah\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\404_airliner\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\541_drum\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\579_grand-piano\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\651_microwave\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\671_mountain-bike\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\732_polaroid\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\949_strawberry\images",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\954_banana\images"]
render_params= [r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\001_goldfish\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\293_cheetah\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\404_airliner\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\541_drum\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\579_grand-piano\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\651_microwave\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\671_mountain-bike\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\732_polaroid\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\949_strawberry\parameters.csv",
                r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_10000\954_banana\parameters.csv"]
bin_sizes = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]



# for i in range(len(target_classes)):
#     target_class = target_classes[i]
#     folder = folder_paths[i]
#     render_param = render_params[i]
#     bin_size = bin_sizes[i]
#     model = "simclr_santi"
#     print(f"Processing target class {target_class} in folder {folder}...")
#     preds, cam, hue, paths, preds_unchanged = outputs_from_folder(
#         folder=folder,
#         render_param=render_param,
#         param_indices=(1, 2),
#         device="cuda",
#         model_name=model,
#     )
#
#     # print distinct predicted classes
#     # print(preds)
#     unique, counts = np.unique(preds, return_counts=True)
#     pairs = list(zip(unique.tolist(), counts.tolist()))
#     top_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
#     print("Predicted class counts (top 10):")
#     imgnet_classes = open(r"C:\Users\silas\PycharmProjects\SimClr_MT\imagenet_classes.txt", "r").read().splitlines()
#     for cls_id, cnt in top_pairs:
#         cls_idx = int(cls_id)
#         cls_name = imgnet_classes[cls_idx] if 0 <= cls_idx < len(imgnet_classes) else "<unknown>"
#         print(f"Index: {cls_name}  - {cnt} images")
#
#     H, e_cam, e_hue = heatmap_from_params(
#         preds=preds,
#         p0=cam,
#         p1=hue,
#         target_class_id=target_class,
#         bins0 = bin_size,
#         bins1 = bin_size
#     )
#
#     plot_heatmap(
#         H, e_cam, e_hue,
#         # title="Camera × Hue",
#         title=f"camera.azimuth x camera.distance - class {imgnet_classes[target_class]})",
#         xlabel=r"$\theta_{camera}$",
#         ylabel=r"$d_{camera}$",
#         # xlabel=r"$\theta_{hue}$",
#         # ylabel=r"$\theta_{camera}$",
#         out_path=f"camera_x_hue_accuracy_{model}_{target_class}_{bin_size}.png",
#     )
#
#     print(preds.shape)

    # santi_plot(target_class, preds_unchanged, cam, hue, outpath=f"camera_x_hue_accuracy_{model}_{target_class}_santi.png")
