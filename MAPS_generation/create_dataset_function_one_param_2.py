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
        # if has_light_param:
        scene.set({'light.hue': 1.5})
        scene.set({'light.power': 1.0})
        scene.set({'background.hue': 1.5})
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

# Unified DICT_SPACES in the same nested format used elsewhere in this file.
DICT_SPACES = {
    # 'camera_azimuth': {
    #     'camera.azimuth': {'circular': True, 'range': [0, 2 * np.pi]}
    # },
    # 'camera_distance': {
    #     'camera.distance': {'range': [2, 8]}
    # },
    # 'camera_elevation': {
    #     'camera.elevation': {'circular': True, 'range': [0, np.pi]}
    # },
    # 'light_power': {
    #     'light.power': {'range': [0, 1]}
    # },
    # 'background_hue': {
    #     'background.hue': {'circular': True, 'range': [0, 2 * np.pi]}
    # }
    'light_azimuth': {
        'light.azimuth': {'circular': True, 'range': [0, 2 * np.pi]}
    }
}

OBJECTS = [
    "banana",
    'acoustic-guitar',
    'african-elephant',
    'airliner',
    'ambulance',
    'banana',
    'goldfish',
    'grand-piano',
    'microwave',
    'monitor',
    'space-shuttle',
    'umbrella',
    "strawberry"
]

dataset = Dataset(root=SCENES_ROOT)

for obj_name in OBJECTS:
    for instance_num in range(1, 6):  # instances 1-5
        scene_id = f'{obj_name}_{instance_num:03d}'
        
        try:
            scene_blend = dataset.get_scene(scene_id).path_blend
            
            for space_name, dict_space in DICT_SPACES.items():
                save_dir = Path('dataset') / obj_name / space_name / f'instance_{instance_num}'
                print(f'\n=== {scene_id} | {space_name} -> {save_dir} ===')
                create_dataset(
                    scene_path=scene_blend,
                    dict_space=dict_space,
                    save_path=save_dir,
                    num_samples=250,
                )
        except Exception as e:
            print(f"Error processing {scene_id}: {e}")
            continue

