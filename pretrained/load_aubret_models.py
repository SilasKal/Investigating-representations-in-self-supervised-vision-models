import re

from torch import nn
from torch.utils import model_zoo
from torchvision.models import resnet50
from torchvision.transforms import transforms

model_urls = {
    'aasimclr': 'https://huggingface.co/aaubret/AASSL/resolve/main/aasimclr.pt',
    'simclr': 'https://huggingface.co/aaubret/AASSL/resolve/main/simclr.pt',
    'simclrtt': 'https://huggingface.co/aaubret/AASSL/resolve/main/simclrtt.pt',
    'cipersimclr': 'https://huggingface.co/aaubret/AASSL/resolve/main/cipersimclr.pt',
}


def mvimgnet(variant):
    model = resnet50()
    model.fc = nn.Identity()
    checkpoint = model_zoo.load_url(model_urls[variant])
    checkpoint = checkpoint["model"]
    new_state_dict = {}
    for k, w in checkpoint.items():
        if re.search("^model.*", k):
            k = ".".join(k.split(".")[1:])
        if re.search("projector.*", k):
            continue
        if re.search("^sup_lin*", k):
            continue
        if "ciper_action_bn" in k:
            continue
        if "predictor" in k:
            continue
        if "action_head" in k:
            continue
        # if "action_projector" in k and "action_projector" in args.keep_proj:
        #     new_k = ".".join(["head_action.layers"] + k.split(".")[2:])
        #     new_state_dict[new_k] = w
        # elif "equivariant_projector" in k and "equivariant_projector" in args.keep_proj:
        #     new_k = ".".join(["head_equivariant.layers"] + k.split(".")[2:])
        #     new_state_dict[new_k] = w
        # elif "equivariant_predictor" in k and "equivariant_predictor" in args.keep_proj:
        #     new_k = ".".join(["head_prediction.layers"] + k.split(".")[2:])
        #     new_state_dict[new_k] = w
        # else:
        new_state_dict[k] = w

    model.load_state_dict(new_state_dict)
    return model

import torch

def mvimgnet_from_pt(ckpt_path, map_location="cpu", weights_only=True):
    model = resnet50()
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    checkpoint = checkpoint["model"]

    new_state_dict = {}
    for k, w in checkpoint.items():
        # strip leading prefixes
        if re.match(r"^model\.", k):
            k = ".".join(k.split(".")[1:])

        # skip non-backbone parts
        if re.match(r"projector\.", k):
            continue
        if re.match(r"sup_lin", k):
            continue
        if "ciper_action_bn" in k:
            continue
        if "predictor" in k:
            continue
        if "action_head" in k:
            continue
        if "action_projector" in k:  # <- NEW: skip those unexpected keys
            continue
        if "equivariant_" in k:       # <- optional: skip other heads
            continue

        new_state_dict[k] = w
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if unexpected:
        print(f"[mvimgnet_from_pt] Unexpected keys: {unexpected}")
    if missing:
        print(f"[mvimgnet_from_pt] Missing keys: {missing}")

    return model



def custom_mvimgnet(filepath):
    model = mvimgnet_from_pt(filepath)
    return model
