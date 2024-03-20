import torch
from torch import float16
from torch.nn import Module


class Perturber(Module):

    def __init__(self, scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def change_noise_scale(self, scale):
        self.scale = scale


class DxPrivacy(Perturber):
    def __init__(self, embedder: Module, vocab_size, epsilon: float = 5.0, *args, **kwargs):
        super().__init__(scale=epsilon, *args, **kwargs)
        self.vocab_size = vocab_size
        self.embedder = embedder

    def forward(self, inputs_embeds):
        if self.scale == 0:
            return inputs_embeds
        with torch.no_grad():
            noise = torch.distributions.laplace.Laplace(0, scale=1 / self.scale).sample(inputs_embeds.size()).to(
                inputs_embeds.device)
            if inputs_embeds.dtype == float16:
                noise = noise.half()
            inputs_embeds = inputs_embeds + noise

        all_words = torch.tensor(list([i for i in range(self.vocab_size)])).to(inputs_embeds.device)
        all_embeds = self.embedder(all_words)
        cosine_similarities = torch.matmul(inputs_embeds, all_embeds.transpose(0, 1))
        max_token = torch.argmax(cosine_similarities, dim=-1)
        return all_embeds[max_token]


class GaussianPerturber(Perturber):

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.size()
        max_values, _ = torch.max(hidden_states.view(batch_size, -1), dim=1, keepdim=True)
        min_values, _ = torch.min(hidden_states.view(batch_size, -1), dim=1, keepdim=True)
        scales = max_values - min_values  # (batch_size, 1)
        noise = torch.randn_like(hidden_states)  # (batch_size, seq_len, hidden_size)
        # multiply the noise  with the scale
        noise = noise * scales.view(batch_size, 1, 1) * self.scale
        res = hidden_states + noise
        return res
