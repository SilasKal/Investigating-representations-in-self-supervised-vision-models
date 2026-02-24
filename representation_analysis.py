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
from sklearn.linear_model import Ridge
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
def extract_layer_embeddings(model, loader, device):
    model.eval()
    model.to(device)

    features = {}
    collected = {}

    # ---- define hooks ----
    def hook(name):
        def fn(module, input, output):
            features[name] = output.detach()
        return fn

    # Register hooks
    model.layer1.register_forward_hook(hook("layer1"))
    model.layer2.register_forward_hook(hook("layer2"))
    model.layer3.register_forward_hook(hook("layer3"))
    model.layer4.register_forward_hook(hook("layer4"))

    # Containers
    collected = {
        "layer1": [],
        "layer2": [],
        "layer3": [],
        "layer4": [],
        "avgpool": []
    }

    with torch.no_grad():
        for images, params in loader:
            images = images.to(device)

            output = model(images)  # forward pass

            # collect spatial layers
            for name in ["layer1", "layer2", "layer3", "layer4"]:
                fmap = features[name]  # shape: (B, C, H, W)
                pooled = F.adaptive_avg_pool2d(fmap, 1).squeeze(-1).squeeze(-1)
                pooled = F.normalize(pooled, dim=1)
                collected[name].append(pooled.cpu())

            # collect final embedding (after avgpool)
            if hasattr(model, "fc"):
                # get avgpool features before fc
                avg_feat = model.avgpool(features["layer4"])
                avg_feat = torch.flatten(avg_feat, 1)
            else:
                avg_feat = output

            avg_feat = F.normalize(avg_feat, dim=1)
            collected["avgpool"].append(avg_feat.cpu())

    # concatenate
    for k in collected:
        collected[k] = torch.cat(collected[k]).numpy()

    return collected, params
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
from pretrained.load_mvimgnet_model import load_mv_model
# for i in ["camera_distance", "camera_elevation", "camera_azimuth", "light_power", "background_hue"]:
for i in [""]:
    import os
    import matplotlib.pyplot as plt
    import pkg_resources
    # instance wise representation analysis (all images from an instance share the same label, e.g. instance ID)

    for index, j in enumerate([r"C:\Users\silas\PycharmProjects\SimClr_MT\finetuning_MAPS_5_instances_all_params\epoch_49.pt",
              r"C:\Users\silas\PycharmProjects\SimClr_MT\training_from_scratch_MAPS_5_instances_all_params\epoch_49.pt",
              r"C:\Users\silas\PycharmProjects\SimClr_MT\supervised_MAPS\sup_resnet50.pt", "V1", "V2"]):
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
                model_path = "supervised_MAPS"
            model, preprocess = load_mv_model(
                j,
                device='cpu')
        #
        # model, preprocess = load_mv_model(r'C:\Users\silas\PycharmProjects\SimClr_MT\finetuning_MAPS_5_instances_all_params\epoch_49.pt', device='cpu')
        # model, preprocess = load_mv_model(r'C:\Users\silas\PycharmProjects\SimClr_MT\training_from_scratch_MAPS_5_instances_all_params\epoch_49.pt', device='cpu')
        # model, preprocess = load_mv_model(r'C:\Users\silas\PycharmProjects\SimClr_MT\supervised_MAPS\sup_resnet50.pt', device='cpu')

        # model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # print('model loaded')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        # BASE_DIR = fr"C:\Users\silas\PycharmProjects\SimClr_MT\strawberry_5_instances_{i}" # adjust to your path
        BASE_DIR = fr"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_new_multiple\umbrella" # adjust to your path
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

        ax.set_title("3D PaCMAP — Instance-wise representation")
        ax.set_xlabel("PaCMAP-1")
        ax.set_ylabel("PaCMAP-2")
        ax.set_zlabel("PaCMAP-3")

        ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{model_path}/pacmap_instance_wise_umbrella_3d.png", dpi=300)
        plt.show()


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