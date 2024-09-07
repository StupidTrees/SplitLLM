from torch import float16
from torch.nn import Linear
from transformers import PreTrainedModel, PretrainedConfig

from sfl.model.reducer.args import DRConfig
from sfl.utils.model import get_embed_size


class DimReduction(PreTrainedModel):
    config_class = DRConfig

    def __init__(self, config: DRConfig, target_config: PretrainedConfig = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if target_config is not None:
            self.config.n_embed = get_embed_size(target_config)
        self.m1 = Linear(self.config.n_embed, self.config.alpha)
        self.m2 = Linear(self.config.alpha, self.config.n_embed)

    def forward(self, x):
        half = x.dtype == float16
        if half:
            x = x.float()

        reduced = self.m1(x)
        recovered = self.m2(reduced)
        return reduced, recovered


