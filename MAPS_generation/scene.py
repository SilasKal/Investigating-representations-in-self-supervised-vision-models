import contextlib
import os
import re
import tempfile
from pathlib import Path

import bpy
from PIL import Image

import transforms
print("Blender:", bpy.app.version_string)
print("Binary:", bpy.app.binary_path)


@contextlib.contextmanager
def suppress_output():
    # Redirect C-level stdout/stderr to os.devnull temporarily
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_out, old_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        os.close(devnull_fd)


class Scene:
    def __init__(self,
                 path_blend,
                 resolution=(224, 224),
                 method='cycles',
                 file_format='PNG',
                 backends=['OPTIX', 'CUDA', 'METAL']):
        self.path_blend = path_blend
        self.object_name = re.sub(r'\.blend$', '', path_blend.name)
        self.resolution = resolution
        self.method = method.lower()
        self.file_format = file_format.upper()
        self.backends = backends
        self.transforms = {}
        self.initialize()

    def get(self):
        return {
            key: transform.get()
            for key, transform in sorted(self.transforms.items())
        }

    def set(self, scene_params):
        for key, value in scene_params.items():
            if key not in self.transforms:
                raise KeyError(f'Unknown transform: {key}')
            self.transforms[key].set(value)

    def render(self, path=None):
        tmp_path = None

        if path is None:
            fd, tmp = tempfile.mkstemp(suffix='.png')
            os.close(fd)              # Blender will write to it
            path = Path(tmp)
            tmp_path = path           # remember to clean up
        else:
            path = Path(path)

        try:
            with suppress_output():
                bpy.context.scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)

            img = Image.open(path).convert('RGB')
            return img

        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.resolve() == self.path_blend.resolve():
            raise RuntimeError(
                f'Trying to overwrite template blend file: {self.path_blend}'
            )

        bpy.ops.wm.save_as_mainfile(filepath=str(path))

    def initialize(self):
        bpy.ops.wm.open_mainfile(filepath=str(self.path_blend))

        for key in transforms.registry.transform_registry.keys():
            self.transforms[key] = transforms.registry.transform_registry[key]()

        scene = bpy.context.scene
        renderer = scene.render
        renderer.resolution_x, renderer.resolution_y = self.resolution
        renderer.image_settings.file_format = self.file_format

        if self.method == 'cycles':
            bpy.context.scene.render.engine = 'CYCLES'
            preferences = bpy.context.preferences
            cycles_preferences = preferences.addons['cycles'].preferences
            world = bpy.context.scene.world
            world.cycles_visibility.diffuse = False
            for backend in self.backends:
                try:
                    cycles_preferences.compute_device_type = backend
                    for dev in cycles_preferences.devices:
                        dev.use = True
                    bpy.context.scene.cycles.device = 'GPU'
                    print(f'Cycles set to GPU via {backend}')
                    break
                except Exception:
                    continue
            else:
                bpy.context.scene.cycles.device = 'CPU'
                print('Cycles set to CPU (no compatible GPU backend found)')
        else:
            try:
                renderer.engine = 'BLENDER_EEVEE'
            except Exception:
                renderer.engine = 'BLENDER_EEVEE_NEXT'

    def __str__(self):
        lines = ['Scene:']
        lines.append(f'  blend file      : {self.path_blend}')
        lines.append(f'  resolution      : {self.resolution[0]} × {self.resolution[1]}')
        lines.append(f'  renderer        : {self.method}')
        lines.append(f'  file format     : {self.file_format}')

        if self.transforms:
            lines.append(f'  transforms ({len(self.transforms)}) :')
            for name in sorted(self.transforms):
                lines.append(f'    - {name}')
        else:
            lines.append('  transforms      : <none>')

        return '\n'.join(lines)
