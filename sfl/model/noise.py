import torch
from torch.nn import Module


class DxPrivacy(Module):
    def __init__(self, embedder: Module, vocab_size, epsilon: float = 5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.epsilon = epsilon
        self.embedder = embedder

    def forward(self, inputs_embeds):
        with torch.no_grad():
            noise = torch.distributions.laplace.Laplace(0, scale=1 / self.epsilon).sample(inputs_embeds.size()).to(
                inputs_embeds.device)
            inputs_embeds = inputs_embeds + noise
        all_words = torch.tensor(list([i for i in range(self.vocab_size)])).to(inputs_embeds.device)
        all_embeds = self.embedder(all_words)
        cosine_similarities = torch.matmul(inputs_embeds, all_embeds.transpose(0, 1))
        max_token = torch.argmax(cosine_similarities, dim=-1)
        return all_embeds[max_token]
