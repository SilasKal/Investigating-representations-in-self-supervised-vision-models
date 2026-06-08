# import clip
import torchvision

from pretrained import load_simclr


class Model:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.layers = []

    def get_layers(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            else:
                self.layers.append(name)


class CLIPResNet50(Model):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load('RN50')
        self.model = self.model.visual
        self.get_layers()


class ResNet50(Model):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        self.model = torchvision.models.resnet50(weights=weights)
        self.preprocess = weights.transforms()
        self.get_layers()


class SimCLRResNet50(Model):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = load_simclr()
        self.get_layers()


MODEL_REGISTRY = {
    'resnet50': ResNet50,
    'clip_resnet50': CLIPResNet50,
    'simclr_resnet50': SimCLRResNet50
}


def get_model(model_name):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]()
    else:
        raise ValueError(f'Unknonw model: {model_name}. '
                         f'Available models: {list(MODEL_REGISTRY.keys())}')
