import dataclasses

import torch
from torch import nn
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from transformers import PretrainedConfig, PreTrainedModel

from sfl.config import AttackerConfig


class AttackModel(PreTrainedModel):
    config_class = AttackerConfig

    def __init__(self, config: AttackerConfig, target_config: PretrainedConfig = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if target_config:
            self.config.n_embed = target_config.n_embd
            self.config.vocab_size = target_config.vocab_size
            name_or_path = target_config.name_or_path
            # if it is a path, use the last dir name
            if '/' in name_or_path:
                if name_or_path.endswith('/'):
                    name_or_path = name_or_path[:-1]
                name_or_path = name_or_path.split('/')[-1]
            self.config.target_model = name_or_path

    def forward(self, x):
        pass


@dataclasses.dataclass
class LSTMAttackerConfig(AttackerConfig):
    hidden_size: int = 256
    dropout: float = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'lstm'


@dataclasses.dataclass
class TransformerAttackerConfig(AttackerConfig):
    dropout: float = 0.1
    num_layers = 2
    nhead = 4
    dim_feedforward = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'trans_decoder'


class LinearAttackModel(AttackModel):
    config_class = AttackerConfig

    def __init__(self, config: AttackerConfig, *args, **kwargs):
        config.model_name = 'linear'
        super().__init__(config, *args, **kwargs)
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        return self.mlp(x)


class LSTMAttackModel(AttackModel):
    config_class = LSTMAttackerConfig

    def __init__(self, config: LSTMAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.lstm = nn.LSTM(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True)
        self.mlp = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden, _ = self.lstm(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=self.config.dropout, train=self.training)
        return self.mlp(hidden)


class GRUAttackModel(AttackModel):
    config_class = LSTMAttackerConfig

    def __init__(self, config: LSTMAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.gru = nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True)
        self.mlp = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden, _ = self.gru(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=self.config.dropout, train=self.training)
        return self.mlp(hidden)


class TransformerDecoderAttackModel(AttackModel):
    config_class = TransformerAttackerConfig

    def __init__(self, config: TransformerAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        layer = TransformerDecoderLayer(d_model=self.config.n_embed, nhead=self.config.nhead,
                                        dim_feedforward=self.config.dim_feedforward,
                                        dropout=self.config.dropout)
        self.decoder = nn.TransformerDecoder(layer, self.config.num_layers, norm=nn.LayerNorm(self.config.n_embed))
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden = self.decoder(x)
        return self.mlp(hidden)


class TransformerEncoderAttackModel(AttackModel):
    config_class = TransformerAttackerConfig

    def __init__(self, config: TransformerAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        layer = TransformerEncoderLayer(d_model=self.config.n_embed, nhead=self.config.nhead,
                                        dim_feedforward=self.config.dim_feedforward,
                                        dropout=self.config.dropout)
        self.encoder = nn.TransformerEncoder(layer, self.config.num_layers, norm=nn.LayerNorm(self.config.n_embed))
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden = self.encoder(x)
        return self.mlp(hidden)
