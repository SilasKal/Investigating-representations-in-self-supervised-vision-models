
import bpy

from .base import BaseTransform
from .registry import register_transform


@register_transform('background.hue')
class BackgroundHue(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['BackgroundMaterial'].node_tree.nodes['HSV'].inputs[0]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value


@register_transform('background.saturation')
class BackgroundSaturation(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['BackgroundMaterial'].node_tree.nodes['HSV'].inputs[1]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value


@register_transform('background.value')
class BackgroundValue(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['BackgroundMaterial'].node_tree.nodes['HSV'].inputs[2]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value


@register_transform('background.noise')
class BackgroundNoise(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['BackgroundMaterial'].node_tree.nodes['Noise'].outputs[0]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value
