#!/usr/bin/python
# _____________________________________________________________________________
import math
# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import sys, os
from typing import Optional, List, Tuple

import torch
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

from torchvision.transforms.v2 import Compose

from utils.constants import DATASETS
from kornia import augmentation as TF
import torchvision


class GaussianBlur(object):
    """
    https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/augmentations.py
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    """
    https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/augmentations.py
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_transform_list(args, crop_size=None, tensor_normalize=True, normalize=None):
    transformations = []
    if args.min_crop != 1 and not args.one_crop and not args.crop_first:
        transformations.append(get_resized_crop(args, crop_size))
    if args.flip:
        transformations.append(get_flip(args))
    if args.jitter != 0 and not args.unijit:
        transformations.append(get_jitter(args))
    if args.grayscale and not args.unijit:
        transformations.append(get_grayscale(args))
    if args.blur:
        transformations.append(TF.RandomGaussianBlur(kernel_size=args.blur, sigma=(0.1, 2.0), p=args.pblur))
    if args.solarize:
        transformations.append(TF.RandomSolarize(p=args.solarize))
    if tensor_normalize:
        transformations.append(normalize)
    return torch.nn.Sequential(*transformations)

def get_transformations(args, crop_size=None, tensor_normalize=True):
    norm_dataset = args.dataset
    normalize = TF.Normalize(mean=DATASETS[norm_dataset]['rgb_mean'], std=DATASETS[norm_dataset]['rgb_std'])
    val_transform = normalize

    if args.contrast != 'time' and args.kornia:
        train_transform = get_transform_list(args, crop_size=crop_size, tensor_normalize=tensor_normalize, normalize=normalize)
    else:
        train_transform = val_transform
    return train_transform, val_transform

def get_resized_crop(args, crop_size):
    ratio = (3.0 / 4, 4 / 3.0)
    crop_size = crop_size
    fn = TF.RandomResizedCrop
    return fn(size=crop_size, scale=(args.min_crop, args.max_crop), ratio=ratio, p=args.pcrop)


def get_jitter(args):
    s = args.jitter_strength
    return TF.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s,p=args.jitter)


def get_grayscale(args):
    return TF.RandomGrayscale(p=0.2)


def get_flip(args):
    return TF.RandomHorizontalFlip(p=args.flip)


def get_action_indices(args):
    """
    Parse action indices from config. Returns a list of indices to use.
    
    Supports:
    - Direct indices: "0,1,4" -> [0, 1, 4]
    - Excluded indices: "^0,1" or via action_exclude_indices
    - Presets for MAPS: "camera" -> all camera params, "azimuth", "distance", "elevation", "roll"
    - Presets for MAPS: "light" -> light params, "background" -> background params
    - Empty string: use all parameters (returns None)
    
    Returns list of indices or None if no filtering should be applied.
    """
    dataset = getattr(args, "dataset", "")
    
    # Get the action indices specification from args
    action_indices_str = getattr(args, "action_indices", "").strip()
    action_exclude_str = getattr(args, "action_exclude_indices", "").strip()
    
    # If no filtering requested, return None (use all indices)
    if not action_indices_str and not action_exclude_str:
        return None
    
    # Define parameter index ranges for different datasets
    PARAMETER_PRESETS = {
        "MAPS": {
            # MAPS has 6 parameters total (camera only):
            # 0-1: azimuth (sin, cos)
            # 2-3: roll (sin, cos)
            # 4: elevation
            # 5: distance (log)
            "camera": [0, 1, 2, 3, 4, 5],  # all camera parameters
            "azimuth": [0, 1],
            "roll": [2, 3],
            "elevation": [4],
            "distance": [5],
            "camera_azimuth": [0, 1],
            "camera_roll": [2, 3],
            "camera_elevation": [4],
            "camera_distance": [5],
            "light": [],  # MAPS doesn't have light params
            "background": [],  # MAPS doesn't have background params
        },
        "MAPSInstanceSplitDataset": {
            # MAPSInstanceSplitDataset has 15 parameters total:
            # 0-1: camera.azimuth (sin, cos)
            # 2-3: camera.roll (sin, cos)
            # 4: camera.elevation
            # 5: camera.distance (log)
            # 6-7: background.hue (sin, cos)
            # 8-10: background.noise, background.saturation, background.value
            # 11-12: light.azimuth (sin, cos)
            # 13: light.elevation
            # 14: light.power
            "camera": [0, 1, 2, 3, 4, 5],  # all camera parameters
            "camera_azimuth": [0, 1],
            "camera_roll": [2, 3],
            "camera_elevation": [4],
            "camera_distance": [5],
            "azimuth": [0, 1],  # camera azimuth
            "roll": [2, 3],  # camera roll
            "elevation": [4],  # camera elevation
            "distance": [5],  # camera distance
            "background": [6, 7, 8, 9, 10],  # all background parameters
            "background_hue": [6, 7],
            "background_noise": [8],
            "background_saturation": [9],
            "background_value": [10],
            "light": [11, 12, 13, 14],  # all light parameters
            "light_azimuth": [11, 12],
            "light_elevation": [13],
            "light_power": [14],
        },
        "CO3D": {
            # CO3D has different structure - implement if needed
            "camera": list(range(9)),  # adjust based on actual CO3D params
        },
    }
    
    # Parse action_indices
    if action_indices_str:
        indices = []
        for spec in action_indices_str.split(","):
            spec = spec.strip()
            if not spec:
                continue
            
            # Check if it's a preset
            if dataset in PARAMETER_PRESETS and spec in PARAMETER_PRESETS[dataset]:
                indices.extend(PARAMETER_PRESETS[dataset][spec])
            else:
                # Try to parse as individual index
                try:
                    idx = int(spec)
                    indices.append(idx)
                except ValueError:
                    raise ValueError(f"Unknown action preset or invalid index: {spec} for dataset {dataset}")
        
        return sorted(list(set(indices)))  # remove duplicates and sort
    
    # Parse action_exclude_indices
    elif action_exclude_str:
        # Get all indices for this dataset
        total_size = get_action_size(args)  # recursive call, but this will return full size
        all_indices = set(range(total_size))
        
        exclude_indices = set()
        for spec in action_exclude_str.split(","):
            spec = spec.strip()
            if not spec:
                continue
            
            # Check if it's a preset
            if dataset in PARAMETER_PRESETS and spec in PARAMETER_PRESETS[dataset]:
                exclude_indices.update(PARAMETER_PRESETS[dataset][spec])
            else:
                try:
                    idx = int(spec)
                    exclude_indices.add(idx)
                except ValueError:
                    raise ValueError(f"Unknown action preset or invalid index: {spec} for dataset {dataset}")
        
        return sorted(list(all_indices - exclude_indices))
    
    return None


def get_action_size(args):
    # 1) manual override wins (lets you change action dims without editing code again)
    if hasattr(args, "action_size"):
        try:
            if int(args.action_size) > 0:
                return int(args.action_size)
        except Exception:
            pass

    # 2) check if we're using filtered indices
    action_indices = get_action_indices_base(args)
    if action_indices is not None:
        return len(action_indices)

    # 3) dataset-specific exceptions
    if getattr(args, "dataset", "") == "CO3D" and not getattr(args, "co3d_quaternion", True):
        return 14

    if getattr(args, "dataset", "") == "MAPS":
        return 6
    
    if getattr(args, "dataset", "") == "MAPSInstanceSplitDataset":
        return 15

    # 4) default from registry
    return DATASETS[args.dataset]["action_size"]


def get_action_indices_base(args):
    """
    Internal helper that doesn't recursively call get_action_size.
    Returns the list of indices or None.
    """
    dataset = getattr(args, "dataset", "")
    action_indices_str = getattr(args, "action_indices", "").strip()
    action_exclude_str = getattr(args, "action_exclude_indices", "").strip()
    
    if not action_indices_str and not action_exclude_str:
        return None
    
    PARAMETER_PRESETS = {
        "MAPS": {
            "camera": [0, 1, 2, 3, 4, 5],
            "azimuth": [0, 1],
            "roll": [2, 3],
            "elevation": [4],
            "distance": [5],
            "camera_azimuth": [0, 1],
            "camera_roll": [2, 3],
            "camera_elevation": [4],
            "camera_distance": [5],
            "light": [],
            "background": [],
        },
        "MAPSInstanceSplitDataset": {
            "camera": [0, 1, 2, 3, 4, 5],
            "camera_azimuth": [0, 1],
            "camera_roll": [2, 3],
            "camera_elevation": [4],
            "camera_distance": [5],
            "azimuth": [0, 1],
            "roll": [2, 3],
            "elevation": [4],
            "distance": [5],
            "background": [6, 7, 8, 9, 10],
            "background_hue": [6, 7],
            "background_noise": [8],
            "background_saturation": [9],
            "background_value": [10],
            "light": [11, 12, 13, 14],
            "light_azimuth": [11, 12],
            "light_elevation": [13],
            "light_power": [14],
        },
        "CO3D": {
            "camera": list(range(9)),
        },
    }
    
    if action_indices_str:
        indices = []
        for spec in action_indices_str.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if dataset in PARAMETER_PRESETS and spec in PARAMETER_PRESETS[dataset]:
                indices.extend(PARAMETER_PRESETS[dataset][spec])
            else:
                try:
                    idx = int(spec)
                    indices.append(idx)
                except ValueError:
                    raise ValueError(f"Unknown action preset or invalid index: {spec}")
        return sorted(list(set(indices)))
    
    elif action_exclude_str:
        # Get total size from dataset constants
        if dataset == "MAPS":
            total_size = 6
        elif dataset == "MAPSInstanceSplitDataset":
            total_size = 15
        elif dataset == "CO3D":
            total_size = 9
        else:
            if dataset in DATASETS:
                total_size = DATASETS[dataset].get("action_size", 0)
            else:
                return None
        
        all_indices = set(range(total_size))
        exclude_indices = set()
        
        for spec in action_exclude_str.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if dataset in PARAMETER_PRESETS and spec in PARAMETER_PRESETS[dataset]:
                exclude_indices.update(PARAMETER_PRESETS[dataset][spec])
            else:
                try:
                    idx = int(spec)
                    exclude_indices.add(idx)
                except ValueError:
                    raise ValueError(f"Unknown action preset or invalid index: {spec}")
        
        return sorted(list(all_indices - exclude_indices))
    
    return None


