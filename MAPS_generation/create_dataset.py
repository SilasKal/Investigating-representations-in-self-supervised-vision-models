from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from dataset import Dataset
from parameters import ParameterSpace, ParameterSweep

dataset = Dataset(root='/projects/EEHPC-DEV-2026D01-092/MAPS_generation/scenes/')
# /work/dldevel/galella/datasets/MAPS/scenes/banana/banana_001.blend
scene = dataset.get_scene('strawberry_005')
print(scene)

params = scene.get()
# params['background.saturation'] = 0.5
# scene.set(params)
# for param, value in params.items():
#     print(f'{param}: {value}')

dict_space = {
    # 'camera.azimuth': {'circular': True, 'range': [0, 2 * np.pi]},
    # 'camera.distance': {'range': [2, 8]},
    # 'camera.elevation': {'range': [0, np.pi ]},
    # 'camera.roll': {'range': [0, 2*np.pi]}, #1
    # 'light.hue' : {'range': [0, np.pi]},
    # 'background.hue' : {'range': [0, 2 * np.pi]},
    'background.noise' : {'range': [0, 1]}, #2
    # 'background.saturation' : {'range': [0, 1]}, #3
    # 'background.value' : {'range': [0, 1]}, #4
    # - camera.fstop
    # 'light.azimuth' : {'circular': True, 'range': [0, 2 * np.pi]}, #9
    # 'light.elevation' : {'circular': True, 'range': [0, np.pi]},#10
    # 'light.power' : {'range': [0, 1]},
}

parameter_space = ParameterSpace(dict_space)
print(parameter_space)

dict_sweep = {
    # 'camera.azimuth': 250,
    # 'camera.distance': 250,
    # 'camera.elevation': 250,
    # "light.hue": 250,
    # 'background.hue': 250,
    # 'camera.roll': 250,
    'background.noise': 250,
    # 'background.saturation': 250,#
    # 'background.value': 250,
    # 'light.azimuth': 250,
    # 'light.elevation': 250,
    # 'light.power': 250,
}


parameter_sweep = ParameterSweep(dict_space, dict_sweep)
for params in parameter_sweep:
    path_dataset = Path().resolve() / Path('dataset') / 'strawberry_background_noise_005'
print(f"Dataset will be saved in {path_dataset}")
path_dataset.mkdir(exist_ok=True, parents=True)

path_images = path_dataset / Path('images')
path_images.mkdir(exist_ok=True, parents=True)

path_csv = path_dataset / Path('parameters.csv')
list_images_names = []

for i, params in tqdm(enumerate(parameter_sweep), total=parameter_sweep.num_images):
    path_image = path_images / f'{i:04d}.png'
    list_images_names.append(path_image.name)
    scene.set(params)
    scene.render(path_image)
df_sweep = parameter_sweep.to_dataframe()
assert len(df_sweep) == len(list_images_names)

df_sweep.insert(0, 'image', list_images_names)
df_sweep.to_csv(path_csv, index=False)