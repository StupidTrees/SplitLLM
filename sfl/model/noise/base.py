from abc import ABC

from torch import nn


class Perturber(nn.Module):
    arg_cls = None

    def __init__(self, scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def change_noise_scale(self, scale):
        self.scale = scale
