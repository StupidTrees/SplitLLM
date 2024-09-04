import torch
from torch import float16

from sfl.config import gaussian_clipping_threshold
from sfl.model.noise.base import Perturber


class GaussianPerturber(Perturber):

    def forward(self, hidden_states):
        if self.scale == 0:
            return hidden_states
        batch_size, seq_len, hidden_size = hidden_states.size()
        scale = torch.max(torch.norm(hidden_states.view(batch_size, -1), p=float('inf'), dim=1, keepdim=True) / gaussian_clipping_threshold,
                          torch.ones(batch_size, 1).to(hidden_states.device))
        if hidden_states.dtype == float16:
            scale = scale.half()
        hidden_states = (hidden_states.view(batch_size, -1) / scale).view(batch_size, seq_len, hidden_size)
        noise = torch.distributions.Laplace(0, 2 / self.scale).sample(hidden_states.size()).to(hidden_states.device)
        if hidden_states.dtype == float16:
            noise = noise.half()
        return hidden_states + noise
