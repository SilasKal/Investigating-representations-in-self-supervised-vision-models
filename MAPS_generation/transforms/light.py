import bpy
import numpy as np

from .base import BaseTransform
from .registry import register_transform


@register_transform('light.radius')
class LightRadius(BaseTransform):
    def __init__(self):
        super().__init__()
        self.light = bpy.data.lights['Light']

    def get(self):
        return float(self.light.shadow_soft_size)

    def set(self, value):
        self.light.shadow_soft_size = value


@register_transform('light.power')
class LightPower(BaseTransform):
    def __init__(self):
        super().__init__()
        self.light = bpy.data.lights['Light']

    def get(self):
        return float(self.light['Power'])

    def set(self, value):
        self.light['Power'] = value
        self.light.update_tag()


@register_transform('light.hue')
class LightHue(BaseTransform):
    def __init__(self):
        super().__init__()
        self.hue = bpy.data.lights['Light'].node_tree.nodes['HSV'].inputs[0]

    def get(self):
        return self.hue.default_value

    def set(self, value):
        self.hue.default_value = value


@register_transform('light.saturation')
class LightSaturation(BaseTransform):
    def __init__(self):
        super().__init__()
        self.saturation = bpy.data.lights['Light'].node_tree.nodes['HSV'].inputs[1]

    def get(self):
        return self.saturation.default_value

    def set(self, value):
        self.saturation.default_value = value


@register_transform('light.azimuth')
class LightAzimuth(BaseTransform):
    def __init__(self):
        super().__init__()
        self.light = bpy.data.objects['Light']

    def get(self):
        return float(self.light['phi'])

    def set(self, value):
        self.light['phi'] = value
        self.light.update_tag()


@register_transform('light.elevation')
class LightElevation(BaseTransform):
    def __init__(self):
        super().__init__()
        self.light = bpy.data.objects['Light']

    def get(self):
        return float(self.light['theta'])

    def set(self, value):
        self.light['theta'] = value
        self.light.update_tag()
