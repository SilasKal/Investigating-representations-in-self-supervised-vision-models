
import bpy

from .base import BaseTransform
from .registry import register_transform


@register_transform('floor.hue')
class FloorHue(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['FloorMaterial'].node_tree.nodes['HSV'].inputs[0]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value


@register_transform('floor.saturation')
class FloorSaturation(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['FloorMaterial'].node_tree.nodes['HSV'].inputs[1]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value


@register_transform('floor.value')
class FloorValue(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['FloorMaterial'].node_tree.nodes['HSV'].inputs[2]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value


@register_transform('floor.alpha')
class FloorAlpha(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['FloorMaterial'].node_tree.nodes['Alpha'].outputs[0]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value


@register_transform('floor.noise')
class FloorNoise(BaseTransform):
    def __init__(self):
        super().__init__()
        self.material = bpy.data.materials['FloorMaterial'].node_tree.nodes['Noise'].outputs[0]

    def get(self):
        return self.material.default_value

    def set(self, value):
        self.material.default_value = value
