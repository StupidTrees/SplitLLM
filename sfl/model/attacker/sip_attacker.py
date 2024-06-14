import dataclasses
import math
import os

import torch
from tokenizers import Tokenizer
from torch import nn, Tensor
from torch.nn import ModuleList
from transformers import PretrainedConfig, PreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from sfl.config import SIPAttackerArguments, DRA_train_label
from sfl.model.attacker.attacker import Attacker
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import required_quantization
from sfl.utils.model import sentence_score_tokens, get_embed_size


class SIPAttacker(Attacker):
    arg_clz = SIPAttackerArguments

    def __init__(self):
        super().__init__()
        self.inverter_b2tr = None
        self.inverter_tr2t = None

    def parse_arguments(self, args, prefix: str):
        res: SIPAttackerArguments = super().parse_arguments(args, prefix)
        if res.dataset is None or len(res.dataset) == 0:
            res.dataset = args.dataset
        if res.target_model_name is None or len(res.target_model_name) == 0:
            res.target_model_name = args.model_name
        if res.target_model_load_bits < 0:
            res.target_model_load_bits = args.load_bits
        res.target_dataset = args.dataset
        res.target_system_sps = args.split_points
        if res.b2tr_layer < 0:
            res.b2tr_layer = int(res.target_system_sps.split('-')[0])
        if res.tr2t_layer < 0:
            res.tr2t_layer = int(res.target_system_sps.split('-')[1])
        if res.model == 'vit':
            res.larger_better = False
        return res

    def load_attacker(self, args, aargs: arg_clz, llm: SplitWrapperModel = None,
                      tokenizer: Tokenizer = None):
        self.inverter_b2tr, self.inverter_tr2t = get_sip_inverter(aargs)

    def attack(self, args, aargs: arg_clz, llm: SplitWrapperModel, tokenizer: Tokenizer,
               simulator: SFLSimulator, batch, b2tr_inter,
               tr2t_inter, all_inters, init=None):
        attacked_result = {}
        encoder_inter = all_inters.get('encoder', None)
        encoder_inter = None if encoder_inter is None else encoder_inter.fx.to(llm.device)
        with torch.no_grad():
            for type, atk in zip(['tr2t', 'b2tr'], [self.inverter_tr2t, self.inverter_b2tr]):
                if atk is None:
                    continue
                if aargs.attack_all_layers:
                    for idx, inter in all_inters.items():
                        if not isinstance(idx, int):
                            continue
                        if llm.type == 'encoder-decoder':
                            attacked = atk(torch.concat([encoder_inter.fx.to(
                                simulator.device), inter.fx.to(atk.device)], dim=1))
                        else:
                            attacked = atk(inter.fx.to(atk.device))
                        attacked_result[f'{type}_{idx}'] = attacked
                else:
                    if type == 'b2tr':
                        if aargs.b2tr_target_layer >= 0:
                            inter = all_inters[aargs.b2tr_target_layer]
                        else:
                            inter = b2tr_inter
                    else:
                        if aargs.tr2t_target_layer >= 0:
                            inter = all_inters[aargs.tr2t_target_layer]
                        else:
                            inter = tr2t_inter
                    if llm.type == 'encoder-decoder':
                        attacked = atk(torch.concat([encoder_inter.fx.to(
                            simulator.device), inter.fx.to(atk.device)], dim=1))
                    else:
                        attacked = atk(inter.fx.to(atk.device))
                    attacked_result[type] = attacked

        return attacked_result


@dataclasses.dataclass
class SIPInverterConfig(PretrainedConfig):
    model_name: str = None
    target_model: str = None
    vocab_size: int = 0
    n_embed: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SIPInverter(PreTrainedModel):
    """
    DRA攻击模型
    """

    config_class = SIPInverterConfig

    def __init__(self, config: SIPInverterConfig, target_config: PretrainedConfig = None, reduce_dim=None, *args,
                 **kwargs):
        super().__init__(config, *args, **kwargs)
        if target_config:
            self.target_config = target_config
            self.config.n_embed = get_embed_size(target_config)
            self.config.vocab_size = target_config.vocab_size
            if reduce_dim:
                self.config.n_embed = reduce_dim
            name_or_path = target_config.name_or_path
            # if it is a path, use the last dir name
            if '/' in name_or_path:
                if name_or_path.endswith('/'):
                    name_or_path = name_or_path[:-1]
                name_or_path = name_or_path.split('/')[-1]
            self.config.target_model = name_or_path

    def forward(self, x) -> Tensor:
        if 'chatglm' in self.config.target_model:
            x = x.permute(1, 0, 2)
        if x.dtype == torch.float16:
            x = x.float()
        return x

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
class LSTMDRAttackerConfig(SIPInverterConfig):
    hidden_size: int = 256
    dropout: float = 0.1
    num_layers = 2
    bidirectional = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'lstm'


@dataclasses.dataclass
class TransformerSIPInverterConfig(SIPInverterConfig):
    dropout: float = 0.1
    hidden_size: int = 256
    num_layers = 1
    nhead = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'trans_decoder'


@dataclasses.dataclass
class TransformerGRUDRAttackerConfig(LSTMDRAttackerConfig, TransformerSIPInverterConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclasses.dataclass
class MOEDRAttackerConfig(SIPInverterConfig):
    dropout: float = 0.1
    hidden_size: int = 256
    expert_scales: list[float] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'moe'
        if self.expert_scales is None:
            self.expert_scales = [0, 10.0, 7.5, 5.0]


class LinearSIPInverter(SIPInverter):
    config_class = SIPInverterConfig

    def __init__(self, config: SIPInverterConfig, *args, **kwargs):
        config.model_name = 'linear'
        super().__init__(config, *args, **kwargs)
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        x = super().forward(x)
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        return self.mlp(x)


class LSTMDRInverter(SIPInverter):
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
        x = super().forward(x)
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden, _ = self.lstm(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=self.config.dropout, train=self.training)
        return self.mlp(hidden)


class GRUDRInverter(SIPInverter):
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
        x = super().forward(x)
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        hidden, _ = self.gru(x)  # hidden [batch_size, seq_len, n_embed]
        hidden = torch.dropout(hidden, p=self.config.dropout, train=self.training)
        return self.mlp(hidden)


class MOEDRInverter(SIPInverter):
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
            if 'chatglm' in self.config.target_model:
                inter = inter.permute(1, 0, 2)
            if inter.dtype == torch.float16:
                inter = inter.float()
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
        x = super().forward(x)
        exp_outputs = [torch.dropout(exp(x)[0], p=self.config.dropout, train=self.training) for exp in self.experts]
        exp_outputs = torch.stack(exp_outputs, dim=1)  # [batch_size, len(experts), seq_len, hidden_size]
        qkv = self.gating_mlp(x)
        gating_hidden, _ = self.gating_attn(qkv, qkv, qkv)  # [batch_size, seq_len, hidden_size]
        gating_hidden = torch.mean(self.gating_mlp2(gating_hidden), dim=1)  # [batch_size, hidden_size]
        weights = torch.softmax(gating_hidden, dim=-1)  # [batch_size, len(experts)]
        output = torch.einsum('besh,be->bsh', exp_outputs, weights)  # [batch_size, seq_len, hidden_size]
        return self.mlp(output)  # [batch_size, seq_len, vocab_size]


class DecoderSIPInverter(SIPInverter):
    config_class = TransformerSIPInverterConfig

    def __init__(self, config: TransformerSIPInverterConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.decoder = ModuleList(
            [GPT2Block(GPT2Config(n_embd=config.n_embed, n_head=config.nhead)) for _ in range(config.num_layers)])
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        x = super().forward(x)
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        for layer in self.decoder:
            x = layer(x)[0]
        return self.mlp(x)


class AttnSIPInverter(SIPInverter):
    config_class = TransformerSIPInverterConfig

    def __init__(self, config: TransformerSIPInverterConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = nn.MultiheadAttention(self.config.n_embed, self.config.nhead, dropout=self.config.dropout)
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        x = super().forward(x)
        x = self.attn(x, x, x)[0]
        return self.mlp(x)


class AttnGRUDRInverter(SIPInverter):
    config_class = TransformerGRUDRAttackerConfig

    def __init__(self, config: TransformerSIPInverterConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = nn.MultiheadAttention(self.config.n_embed, self.config.nhead, dropout=self.config.dropout)
        # self.decoder = GPT2Block(GPT2Config(n_embd=config.n_embed, n_head=config.nhead))
        self.gru = nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True,
                          bidirectional=config.bidirectional)
        hidden_size = self.config.hidden_size
        if config.bidirectional:
            hidden_size *= 2
        self.mlp = nn.Linear(hidden_size, self.config.vocab_size)

    def forward(self, x):
        x = super().forward(x)
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        x = self.attn(x, x, x)[0]  # self.decoder(x)[0]
        x, _ = self.gru(x)  # hidden [batch_size, seq_len, n_embed]
        x = torch.dropout(x, p=self.config.dropout, train=self.training)
        return self.mlp(x)


class GRUAttnSIPInverter(SIPInverter):
    config_class = TransformerGRUDRAttackerConfig

    def __init__(self, config: TransformerGRUDRAttackerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.gru = nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True,
                          bidirectional=config.bidirectional)
        hidden_size = self.config.hidden_size
        if config.bidirectional:
            hidden_size *= 2
        # self.mlp0 = nn.Linear(self.config.n_embed, hidden_size)
        # self.decoder = GPT2Block(GPT2Config(n_embd=hidden_size, n_head=config.nhead))
        self.attn = nn.MultiheadAttention(self.config.n_embed, self.config.nhead, dropout=self.config.dropout,
                                          kdim=hidden_size,
                                          vdim=hidden_size)
        self.mlp = nn.Linear(self.config.n_embed, self.config.vocab_size)

    def forward(self, x):
        x = super().forward(x)
        # x[batch_size, seq_len, n_embed]
        # output [batch_size,seq_len, vocab_size]
        gru_x, _ = self.gru(x)  # hidden [batch_size, seq_len, n_embed]
        # mlp_x = self.mlp0(x)
        gru_x = torch.dropout(gru_x, p=self.config.dropout, train=self.training)
        out = self.attn(x, gru_x, gru_x)[0]
        return self.mlp(out)


# class GRUCrossAttnDRAttacker(DRAttacker):
#     config_class = TransformerGRUDRAttackerConfig
#
#     def __init__(self, config: TransformerGRUDRAttackerConfig, *args, **kwargs):
#         super().__init__(config, *args, **kwargs)
#         self.gru = nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True,
#                           bidirectional=config.bidirectional)
#         hidden_size = self.config.hidden_size
#         if config.bidirectional:
#             hidden_size *= 2
#         self.decoder = GPT2Block(GPT2Config(n_embd=hidden_size, n_head=config.nhead))
#         self.mlp = nn.Linear(hidden_size, self.config.vocab_size)
#
#     def forward(self, x):
#         x = super().forward(x)
#         # x[batch_size, seq_len, n_embed]
#         # output [batch_size,seq_len, vocab_size]
#         x, _ = self.gru(x)  # hidden [batch_size, seq_len, n_embed]
#         x = torch.dropout(x, p=self.config.dropout, train=self.training)
#         x = self.decoder(x)[0]
#         return self.mlp(x)


@dataclasses.dataclass
class ViTDRAttackerConfig(SIPInverterConfig):
    hidden_size: int = 512
    dropout: float = 0.1
    bidirectional = False

    patch_num = 0
    patch_size = 16
    image_size = 0
    output_channels = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'gru '


class ViTDRAttacker(PreTrainedModel):
    config_class = ViTDRAttackerConfig

    def __init__(self, config: ViTDRAttackerConfig, target_config: PretrainedConfig = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        if target_config:
            name_or_path = target_config.name_or_path
            if '/' in name_or_path:
                if name_or_path.endswith('/'):
                    name_or_path = name_or_path[:-1]
                name_or_path = name_or_path.split('/')[-1]
            self.config.target_model = name_or_path
            self.config.n_embed = target_config.hidden_size
            self.config.image_size = target_config.image_size
            self.config.patch_size = target_config.patch_size
            self.config.patch_num = target_config.image_size ** 2 // target_config.patch_size ** 2 + 1
        # GRU layer
        self.gru = nn.GRU(input_size=self.config.n_embed, hidden_size=self.config.hidden_size, batch_first=True)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config.hidden_size,
                               out_channels=128,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=5,
                               stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=3,
                               kernel_size=5,
                               stride=2,
                               padding=2, output_padding=1),
        )

    def forward(self, x):
        # x: (batch_size, patch_num, hidden_size)
        batch_size, patch_num, n_embed = x.shape
        x = x[:, :-1, :]  # 去除最后一个填充token，该维度可被开方
        # 使用GRU处理输入
        x, _ = self.gru(x)  # (batch_size, patch_num-1, hidden_size)
        hidden_size = x.shape[-1]
        x = torch.dropout(x, self.config.dropout, self.training)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, hidden_size, patch_num)
        # 重塑特征图以适应反卷积层
        feature_map_size = int(math.sqrt(patch_num))
        x = x.view(batch_size, hidden_size, feature_map_size, feature_map_size)
        # 通过反卷积层生成图片
        # print(x.shape)
        x = self.conv_transpose(x)  # (batch_size, 3, 224, 224)
        # x = self.conv(x)
        # normalize to -1,1
        x = torch.tanh(x)
        return x


def get_inverter_class(attack_model):
    attacker_cls = LSTMDRInverter
    if attack_model == 'lstm':
        attacker_cls = LSTMDRInverter
    elif attack_model == 'gru':
        attacker_cls = GRUDRInverter
    elif attack_model == 'linear':
        attacker_cls = LinearSIPInverter
    elif attack_model == 'dec':
        attacker_cls = DecoderSIPInverter
    elif attack_model == 'moe' or attack_model == 'moe2':
        attacker_cls = MOEDRInverter
    elif attack_model == 'vit':
        attacker_cls = ViTDRAttacker
    elif attack_model == 'attngru':
        attacker_cls = AttnGRUDRInverter
    elif attack_model == 'gruattn':
        attacker_cls = GRUAttnSIPInverter
    elif attack_model == 'attn':
        attacker_cls = AttnSIPInverter
    return attacker_cls


def get_sip_inverter(dra_config: SIPAttackerArguments):
    if not dra_config.b2tr_enable and not dra_config.tr2t_enable:
        return None, None
    dataset = dra_config.dataset
    if dataset is None:
        dataset = dra_config.target_dataset

    prefix = dra_config.prefix
    model_name = dra_config.target_model_name
    if required_quantization(model_name):
        model_name += f"-{dra_config.target_model_load_bits}bits"
    attacker_path = dra_config.path + f'{model_name}/{dataset}/'
    print(attacker_path)
    matches = []
    for d in os.listdir(attacker_path):
        pattern = f'{DRA_train_label[dataset]}*{dra_config.train_frac:.3f}'
        if ',' in dra_config.dataset:
            pattern = f'Tr{dra_config.train_frac:.3f}'
        if d.startswith(pattern):
            attacker_path = os.path.join(attacker_path, d) + '/'
            matches.append(attacker_path)
    assert len(matches) > 0
    attacker_path_1 = None
    attacker_path_2 = None
    for attacker_path in matches:
        attacker_path += f'{dra_config.model}'
        if dra_config.b2tr_enable and attacker_path_1 is None:
            sp1 = int(dra_config.target_system_sps.split('-')[0])
            if dra_config.b2tr_layer >= 0:
                sp1 = dra_config.b2tr_layer
            attacker_path_1 = attacker_path + f'/layer{sp1}/' + prefix
            l = sorted(list(os.listdir(attacker_path_1)), key=lambda x: float(x.split('_')[-1]))[
                -1 if dra_config.larger_better else 0]
            attacker_path_1 = os.path.join(attacker_path_1, l)
            if not os.path.exists(attacker_path_1):
                attacker_path_1 = None

        if dra_config.tr2t_enable and attacker_path_2 is None:
            sp2 = int(dra_config.target_system_sps.split('-')[1])
            if dra_config.tr2t_layer >= 0:
                sp2 = dra_config.tr2t_layer
            attacker_path_2 = attacker_path + f'/layer{sp2}/' + prefix
            if not os.path.exists(attacker_path_2):
                attacker_path_2 = attacker_path_2.replace('tr2t', 'b2tr')
            l = sorted(list(os.listdir(attacker_path_2)), key=lambda x: float(x.split('_')[-1]))[-1]
            attacker_path_2 = os.path.join(attacker_path_2, l)
            if not os.path.exists(attacker_path_2):
                attacker_path_2 = None

    attacker, attacker2 = None, None
    if attacker_path_1:
        attacker = get_inverter_class(dra_config.model).from_pretrained(attacker_path_1)
    if attacker_path_2:
        attacker2 = get_inverter_class(dra_config.model).from_pretrained(attacker_path_2)
    return attacker, attacker2
