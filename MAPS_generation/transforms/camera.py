import bpy
import numpy as np
from mathutils import Quaternion, Vector

from .base import BaseTransform
from .registry import register_transform


@register_transform('camera.distance')
class CameraDistance(BaseTransform):
    def __init__(self):
        super().__init__()
        self.camera_parent = bpy.data.objects['CameraParent']

    def get(self):
        return self.camera_parent['r']

    def set(self, value):
        self.camera_parent['r'] = value
        self.camera_parent.update_tag()


@register_transform('camera.elevation')
class CameraElevation(BaseTransform):
    def __init__(self):
        super().__init__()
        self.camera_parent = bpy.data.objects['CameraParent']

    def get(self):
        return self.camera_parent['theta']

    def set(self, value):
        self.camera_parent['theta'] = value
        self.camera_parent.update_tag()


@register_transform('camera.azimuth')
class CameraAzimuth(BaseTransform):
    def __init__(self):
        super().__init__()
        self.camera_parent = bpy.data.objects['CameraParent']

    def get(self):
        return self.camera_parent['phi']

    def set(self, value):
        self.camera_parent['phi'] = value
        self.camera_parent.update_tag()


@register_transform('camera.roll')
class CameraRoll(BaseTransform):
    def __init__(self):
        super().__init__()
        self.camera = bpy.data.objects['Camera']

    def get(self):
        return self.camera.rotation_euler[2]

    def set(self, value):
        self.camera.rotation_euler[2] = value


@register_transform('camera.fstop')
class CameraFocus(BaseTransform):
    def __init__(self):
        super().__init__()
        self.camera = bpy.data.cameras['Camera']
        self.camera.dof.use_dof = True

    def get(self):
        return self.camera.dof.aperture_fstop

    def set(self, value):
        self.camera.dof.aperture_fstop = value
