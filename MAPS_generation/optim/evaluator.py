import torch


class Objective:
    def __call__(self, logits, class_id):
        raise NotImplementedError


class NeuronObjective:
    def __call__(self, activations):
        raise NotImplementedError


class ChannelMeanObjective(NeuronObjective):
    def __init__(self, channel):
        self.channel = channel

    def __call__(self, activations):
        return activations[:, self.channel].mean(dim=(1, 2))


class SpatialNeuronObjective(NeuronObjective):
    def __init__(self, channel, i, j):
        self.channel = channel
        self.i = i
        self.j = j

    def __call__(self, activations):
        return activations[:, self.channel, self.i, self.j]


class LogSoftmaxObjective(Objective):
    def __call__(self, logits, class_id):
        logit_class = logits[:, class_id]
        sum_exp = torch.logsumexp(logits, dim=1)
        return logit_class - sum_exp


class ModelEvaluator:
    def __init__(self, model, preprocess, objective, class_id, device):
        self.model = model
        self.preprocess = preprocess
        self.objective = objective
        self.class_id = class_id
        self.device = device

    @torch.no_grad()
    def evaluate(self, images):
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        logits = self.model(x)
        scores = self.objective(logits, self.class_id)
        probs = torch.softmax(logits, dim=1)
        return scores.cpu().numpy(), probs.cpu().numpy()


class NeuronEvaluator:
    def __init__(self, model, preprocess, layer, idx, objective, class_id, device):
        self.model = model
        self.preprocess = preprocess
        self.layer = layer
        self.idx = idx
        self.objective = objective
        self.class_id = class_id
        self.device = device
        self.activations = None

        def hook(_, __, output):
            self.activations = output.detach()

        self._hook_handle = self.layer.register_forward_hook(hook)

    @torch.no_grad()
    def evaluate(self, images):
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        logits = self.model(x)
        scores = self.objective(self.activations)
        probs = torch.softmax(logits, dim=1)
        return scores.cpu().numpy(), probs.cpu().numpy()
    