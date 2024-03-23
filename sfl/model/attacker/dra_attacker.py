import dataclasses
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer, ModuleList
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from sfl.config import DRAttackerConfig
from sfl.utils.model import sentence_score_tokens


class DRAttacker(PreTrainedModel):
    """
    DRA攻击模型
    """
    config_class = DRAttackerConfig

    def __init__(self, config: DRAttackerConfig, target_config: PretrainedConfig = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if target_config:
            self.target_config = target_config
            if hasattr(target_config, 'n_embd'):
                self.config.n_embed = target_config.n_embd
            elif hasattr(target_config, 'hidden_size'):
                self.config.n_embed = target_config.hidden_size
            elif hasattr(target_config, 'd_model'):
                self.config.n_embed = target_config.d_model
            self.config.vocab_size = target_config.vocab_size
            name_or_path = target_config.name_or_path
            # if it is a path, use the last dir name
            if '/' in name_or_path:
                if name_or_path.endswith('/'):
                    name_or_path = name_or_path[:-1]
                name_or_path = name_or_path.split('/')[-1]
            self.config.target_model = name_or_path

    def forward(self, x) -> Tensor:
        pass

    def search(self, x, base_model, beam_size=6):
        logits = self.forward(x.to(self.device))
        batch_size, seq_len, vocab_size = logits.shape
        beams = [(None, [0] * batch_size)] * beam_size
        for step in range(seq_len):
            candidates = []
            for sentence_batch, sent_score_batch in beams:
                last_token_logits = logits[:, step, :]
                topk_probs, topk_indices = torch.topk(last_token_logits, beam_size)
                topk_probs = torch.softmax(topk_probs, dim=-1)
                for k in range(beam_size):
                    prob, token = topk_probs[:, k].unsqueeze(-1), topk_indices[:, k].unsqueeze(-1)  # (batch_size, 1)
                    sents = torch.cat([sentence_batch, token],
                                      dim=1) if sentence_batch is not None else token  # (batch_size, seq++)
                    candidate_score = sentence_score_tokens(sents, base_model).unsqueeze(-1)  # (batch_size, 1)
                    score = prob * 5 - candidate_score
                    # print(prob.shape, candidate_score.shape, score.shape)
                    candidates.append((sents, score))
            new_list = []
            for batch in range(batch_size):
                # print(candidates)
                candidates_batch = [(c[batch, :].unsqueeze(0), score[batch, :].unsqueeze(0)) for c, score in
                                    candidates]
                # print(candidates_batch)
                candidates_batch = sorted(candidates_batch, key=lambda x: x[-1], reverse=True)
                if len(new_list) == 0:
                    new_list = candidates_batch
                else:
                    nl = []
                    for (sent, score), (sent2, score2) in zip(new_list, candidates_batch):
                        nl.append((torch.concat([sent, sent2], dim=0), torch.concat([score, score2], dim=0)))
                    new_list = nl
            beams = new_list[:beam_size]
        return beams[0][0]


@dataclasses.dataclass
class LSTMDRAttackerConfig(DRAttackerConfig):
    hidden_size: int = 256
    dropout: float = 0.1
    bidirectional = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'lstm'


@dataclasses.dataclass
class TransformerDRAttackerConfig(DRAttackerConfig):
    dropout: float = 0.1
    num_layers = 2
    nhead = 4
    dim_feedforward = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'trans_decoder'


@dataclasses.dataclass
class MOEDRAttackerConfig(DRAttackerConfig):
    dropout: float = 0.1
    hidden_size: int = 256
    expert_scales: list[float] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'moe'
        if self.expert_scales is None:
            self.expert_scales = [0, 10.0, 7.5, 5.0]




class LinearDRAttacker(DRAttacker):
    config_class = DRAttackerConfig

    def __init__(self, config: DRAttackerConfig, *args, **kwargs):
        config.model_name = 'linear'
        super().__init__(config, *args, **kwargs)
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        return self.mlp(x)


class LSTMDRAttacker(DRAttacker):
    config_class = LSTMDRAttackerConfig

    def __init__(self, config: LSTMDRAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        hidden_size = self.config.hidden_size
        if config.bidirectional:
            hidden_size *= 2
        self.lstm = nn.LSTM(input_size=self.config.n_embed, hidden_size=hidden_size, batch_first=True,
                            bidirectional=config.bidirectional)
        self.mlp = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden, _ = self.lstm(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=self.config.dropout, train=self.training)
        return self.mlp(hidden)


class GRUDRAttacker(DRAttacker):
    config_class = LSTMDRAttackerConfig

    def __init__(self, config: LSTMDRAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.gru = nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True,
                          bidirectional=config.bidirectional)
        hidden_size = self.config.hidden_size
        if config.bidirectional:
            hidden_size *= 2
        self.mlp = nn.Linear(hidden_size, self.config.vocab_size)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden, _ = self.gru(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=self.config.dropout, train=self.training)
        return self.mlp(hidden)


class MOEDRAttacker(DRAttacker):
    config_class = MOEDRAttackerConfig

    def __init__(self, config: MOEDRAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.experts = ModuleList(
            [nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True)
             for _ in config.expert_scales])
        # self.gating_rnn = nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size,
        #                          batch_first=True)
        # self.gating_mlp = nn.Linear(self.config.n_embed, self.config.hidden_size//2)
        self.gating_mlp = nn.Linear(self.config.n_embed, self.config.hidden_size)
        self.gating_mlp2 = nn.Linear(self.config.hidden_size, len(config.expert_scales))
        self.gating_attn = nn.MultiheadAttention(self.config.hidden_size, 4, dropout=self.config.dropout)
        # self.gating_rnn = nn.GRU(
        #     input_size=self.config.n_embed + len(config.expert_scales) * self.config.hidden_size,
        #     hidden_size=config.hidden_size//2, batch_first=True)

        self.mlp = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def train_exp_forward(self, inters: list):
        assert self.training
        assert len(inters) == len(self.experts)
        outputs = []
        for inter, exp in zip(inters, self.experts):
            if inter is None:
                outputs.append(None)
                continue
            hidden = torch.dropout(exp(inter)[0], p=self.config.dropout, train=self.training)
            outputs.append(self.mlp(hidden))
        return outputs

    def freeze_parts(self, experts=False, freeze=True):
        if experts:
            # self.mlp.requires_grad_(not freeze)
            for expert in self.experts:
                expert.requires_grad_(not freeze)
        else:
            self.gating_attn.requires_grad_(not freeze)
            self.gating_mlp.requires_grad_(not freeze)
            self.gating_mlp2.requires_grad_(not freeze)

    def forward(self, x) -> Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        exp_outputs = [torch.dropout(exp(x)[0], p=self.config.dropout, train=self.training) for exp in self.experts]
        exp_outputs = torch.stack(exp_outputs, dim=1)  # [batch_size, len(experts), seq_len, hidden_size]
        qkv = self.gating_mlp(x)
        gating_hidden, _ = self.gating_attn(qkv, qkv, qkv)  # [batch_size, seq_len, hidden_size]
        gating_hidden = torch.mean(self.gating_mlp2(gating_hidden), dim=1)  # [batch_size, hidden_size]
        weights = torch.softmax(gating_hidden, dim=-1)  # [batch_size, len(experts)]
        output = torch.einsum('besh,be->bsh', exp_outputs, weights)  # [batch_size, seq_len, hidden_size]
        return self.mlp(output)  # [batch_size, seq_len, vocab_size]


class TransformerDecoderDRAttacker(DRAttacker):
    config_class = TransformerDRAttackerConfig

    def __init__(self, config: TransformerDRAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        layer = TransformerDecoderLayer(d_model=self.config.n_embed, nhead=self.config.nhead,
                                        dim_feedforward=self.config.dim_feedforward,
                                        dropout=self.config.dropout)
        self.decoder = ModuleList([GPT2Block(self.target_config, i) for i in range(
            self.config.num_layers)])  # nn.TransformerDecoder(layer, self.config.num_layers, norm=nn.LayerNorm(self.config.n_embed))
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        for md in self.decoder:
            x = md(x)[0]
        return self.mlp(x)


class TransformerEncoderDRAttacker(DRAttacker):
    config_class = TransformerDRAttackerConfig

    def __init__(self, config: TransformerDRAttackerConfig, *args, **kwargs):
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
