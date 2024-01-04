import torch
from torch import nn
from torch.nn import Module
from transformers import PretrainedConfig, GPT2Config, PreTrainedModel


class AttackModel(PreTrainedModel):

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def forward(self, x):
        pass


class GPT2AttackModel(AttackModel):

    def __init__(self, config: GPT2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.hidden_size = 256
        self.lstm = nn.LSTM(input_size=config.n_embd, hidden_size=self.hidden_size, batch_first=True)
        self.mlp = nn.Linear(self.hidden_size, config.vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden, _ = self.lstm(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=0.1, train=self.training)
        return self.mlp(hidden)
