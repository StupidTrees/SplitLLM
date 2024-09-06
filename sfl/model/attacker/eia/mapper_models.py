from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import Linear
from transformers import PreTrainedModel, PretrainedConfig

from sfl.model.llm.glm.configuration_chatglm import ChatGLMConfig
from sfl.model.llm.glm.modeling_chatglm import MLP



class TakeFirst(nn.Module):
    def forward(self, x):
        return x[0]



@dataclass
class LMMapperConfig(PretrainedConfig):
    n_embed: int = 0
    n_layers: int = 1
    structure: str = 'linear'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LMMapper(PreTrainedModel):
    config_class = LMMapperConfig

    def __init__(self, config: LMMapperConfig, target_config: PretrainedConfig = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if target_config:
            self.target_config = target_config
            if hasattr(target_config, 'n_embd'):
                self.config.n_embed = target_config.n_embd
            elif hasattr(target_config, 'hidden_size'):
                self.config.n_embed = target_config.hidden_size
            elif hasattr(target_config, 'd_model'):
                self.config.n_embed = target_config.d_model
            name_or_path = target_config.name_or_path
            # if it is a path, use the last dir name
            if '/' in name_or_path:
                if name_or_path.endswith('/'):
                    name_or_path = name_or_path[:-1]
                name_or_path = name_or_path.split('/')[-1]
            self.config.target_model = name_or_path
        # self.mapper = Linear(config.n_embed, self.config.n_embed)
        self.mapper = torch.nn.Sequential()
        hidden_size = config.n_embed
        if config.structure == 'linear':
            for i in range(config.n_layers):
                if config.n_layers > 1 and i != config.n_layers - 1:
                    self.mapper.add_module(f'linear_{i}', Linear(config.n_embed, hidden_size))
                    self.mapper.add_module(f'activation_{i}', torch.nn.SiLU())
                elif config.n_layers > 1 and i == config.n_layers - 1:
                    self.mapper.add_module(f'linear_{i}', Linear(hidden_size, config.n_embed))
                else:
                    self.mapper.add_module(f'linear_{i}', Linear(config.n_embed, config.n_embed))
        elif config.structure == 'glm':
            self.mapper.add_module(f'mlp',
                                   MLP(ChatGLMConfig(hidden_size=config.n_embed, ffn_hidden_size=hidden_size // 2)))
        elif config.structure == 'gru':
            self.mapper.add_module('gru', torch.nn.GRU(config.n_embed, 512, batch_first=True))
            self.mapper.add_module('tf', TakeFirst())
            self.mapper.add_module('relu', torch.nn.SiLU())
            self.mapper.add_module('mlp', torch.nn.Linear(512, config.n_embed))

    def forward(self, hidden_states):
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()
        return self.mapper(hidden_states)
