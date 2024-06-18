import torch
from torch import float16

from sfl.model.noise.base import Perturber


class GaussianPerturber(Perturber):

    def forward(self, hidden_states):
        if self.scale == 0:
            return hidden_states
        batch_size, seq_len, hidden_size = hidden_states.size()
        # clip the hidden_states by x=x/scale, scale=max(1, x.inf_norm/G), scale:(batch_size, 1, 1)
        G = 2000
        scale = torch.max(torch.norm(hidden_states.view(batch_size, -1), p=float('inf'), dim=1, keepdim=True) / G,
                          torch.ones(batch_size, 1).to(hidden_states.device))
        if hidden_states.dtype == float16:
            scale = scale.half()
        hidden_states = (hidden_states.view(batch_size, -1) / scale).view(batch_size, seq_len, hidden_size)
        noise = torch.distributions.Laplace(0, 2 / self.scale).sample(hidden_states.size()).to(hidden_states.device)
        if hidden_states.dtype == float16:
            noise = noise.half()
        return hidden_states + noise
