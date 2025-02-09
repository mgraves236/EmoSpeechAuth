import torch

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.normal(self.mean, self.std, size=tensor.size()).to(tensor.device)


class EmbeddingDropout:
    def __init__(self, dropout_prob=0.5):
        self.dropout_prob = dropout_prob

    def __call__(self, tensor):
        mask = (torch.rand(tensor.size(), device=tensor.device) > self.dropout_prob).float()
        return tensor * mask


class EmbeddingScale:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, tensor):
        scale_factor = torch.FloatTensor(1).uniform_(*self.scale_range).to(tensor.device)
        return tensor * scale_factor