import pylab as p
from torch.optim import Adam, AdamW

from sfl.model.noise.base import Perturber
from sfl.utils.model import dist_corr, ParamRestored


class NoPeekSimulatedPerturber(Perturber):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = None

    def store_embedding(self, embedding):
        self.embedding = embedding.clone().detach()

    def forward(self, hidden_states):
        return hidden_states
