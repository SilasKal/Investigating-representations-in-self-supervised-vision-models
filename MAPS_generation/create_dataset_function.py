from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from dataset import Dataset
from parameters import ParameterSpace, ParameterSweep
from scene import Scene


def create_dataset(
    scene_path: str | Path,
    dict_space: dict,
    save_path: str | Path,
    num_samples: int = 250,
):
    """Generate a parameter-sweep dataset for a single scene.

    Parameters
    ----------
    scene_path : str | Path
        Path to a ``.blend`` file **or** a scene identifier that can be
        resolved by :class:`Dataset` (e.g. ``"strawberry_005"``).
    dict_space : dict
        Parameter-space definition, e.g.
        ``{'camera.roll': {'range': [-0.78, 0.78]}}``.
    save_path : str | Path
        Directory where the dataset (images + ``parameters.csv``) will be
        saved.
    num_samples : int, optional
        Number of linearly-spaced samples per parameter (default 250).
    """
    # --- resolve scene -------------------------------------------------------
    scene_path = Path(scene_path)
    if scene_path.suffix == '.blend':
        scene = Scene(path_blend=scene_path)
    else:
        # Treat as a root directory containing class sub-folders
        raise ValueError(
            "scene_path must point to a .blend file. "
            f"Got: {scene_path}"
        )

    print(scene)
    print(ParameterSpace(dict_space))

    # --- build sweep ---------------------------------------------------------
    dict_sweep = {name: num_samples for name in dict_space}
    parameter_sweep = ParameterSweep(dict_space, dict_sweep)

    # --- prepare output dirs -------------------------------------------------
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    path_images = save_path / 'images'
    path_images.mkdir(exist_ok=True, parents=True)

    path_csv = save_path / 'parameters.csv'

    # --- render --------------------------------------------------------------
    has_light_param = any(k.startswith('light.') for k in dict_space)

    list_image_names = []
    for i, params in tqdm(enumerate(parameter_sweep), total=parameter_sweep.num_images):
        path_image = path_images / f'{i:04d}.png'
        list_image_names.append(path_image.name)
        scene.set(params)
        if has_light_param:
            scene.set({'background.saturation': 0.5})
        scene.render(path_image)

    # --- save csv ------------------------------------------------------------
    df_sweep = parameter_sweep.to_dataframe()
    assert len(df_sweep) == len(list_image_names)

    df_sweep.insert(0, 'image', list_image_names)
    df_sweep.to_csv(path_csv, index=False)
    print(f"Dataset saved in {save_path}")

    return df_sweep


# ---------------------------------------------------------------------------
# Example: run from the command line
# ---------------------------------------------------------------------------
SCENES_ROOT = Path('/projects/EEHPC-DEV-2026D01-092/MAPS_generation/scenes/')

DICT_SPACES = {
    'camera_roll':              {'camera.roll':              {'range': [0, 2 * np.pi]}},
    'background_noise':         {'background.noise':         {'range': [0, 1]}},
    'background_saturation':    {'background.saturation':    {'range': [0, 1]}},
    'background_value':         {'background.value':         {'range': [0, 1]}},
    'light_azimuth':            {'light.azimuth':            {'circular': True, 'range': [0, 2 * np.pi]}},
    'light_elevation':          {'light.elevation':          {'circular': True, 'range': [0, np.pi]}},
}

# SCENE_IDS = [
#     'strawberry_001',
#     'strawberry_002',
#     'strawberry_003',
#     'strawberry_004',
#     # 'strawberry_005',
# ]
SCENE_IDS = [
    'banana_001',
    'banana_002',
    'banana_003',
    'banana_004',
    'banana_005',
]

dataset = Dataset(root=SCENES_ROOT)

for scene_id in SCENE_IDS:
    scene_blend = dataset.get_scene(scene_id).path_blend
    instance_num = scene_id.split('_')[-1]          # e.g. "005"

    for space_name, dict_space in DICT_SPACES.items():
        save_dir = Path('dataset') / f'banana_{space_name}_{instance_num}'
        print(f'\n=== {scene_id} | {space_name} -> {save_dir} ===')
        create_dataset(
            scene_path=scene_blend,
            dict_space=dict_space,
            save_path=save_dir,
            num_samples=250,
        )
