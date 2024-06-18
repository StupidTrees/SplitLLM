import os
from dataclasses import dataclass
from typing import Any

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import Linear
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel

from sfl.config import DRA_train_label, mapper_path
from sfl.model.attacker.base import Attacker
from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator, ParamRestored
from sfl.utils.exp import required_quantization
from sfl.utils.model import evaluate_attacker_rouge, FLConfigHolder, \
    get_embedding_matrix


@dataclass
class LMMapperConfig(PretrainedConfig):
    n_embed: int = 0
    n_layers: int = 1
    structure: str = 'linear'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TakeFirst(nn.Module):
    def forward(self, x):
        return x[0]


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
        elif config.structure == 'gru':
            self.mapper.add_module('gru', torch.nn.GRU(config.n_embed, 512, batch_first=True))
            self.mapper.add_module('tf', TakeFirst())
            self.mapper.add_module('relu', torch.nn.SiLU())
            self.mapper.add_module('mlp', torch.nn.Linear(512, config.n_embed))

    def forward(self, hidden_states):
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()
        return self.mapper(hidden_states)


@dataclass
class EIAArguments:
    enable: bool = False
    at: str = 'b2tr'
    mapper_target_model_name: str = None
    mapper_target_model_load_bits: int = -1
    mapper_train_frac: float = 1.0
    mapper_path: str = mapper_path
    mapper_dataset: str = ''
    mapper_targets: str = None
    epochs: int = 72000
    lr: float = 0.09
    wd: float = 0.01
    temp: float = 0.2
    cross_model: str = None


def get_eia_mapper(aarg: EIAArguments):
    dataset = aarg.mapper_dataset
    model_name = aarg.mapper_target_model_name
    if required_quantization(model_name):
        model_name += f"-{aarg.mapper_target_model_load_bits}bits"
    mapper_path = aarg.mapper_path + f'{model_name}/{dataset}/'
    matches = []
    for d in os.listdir(mapper_path):
        pattern = f'{DRA_train_label[dataset]}*{aarg.mapper_train_frac:.3f}'
        if ',' in aarg.mapper_dataset:
            pattern = f'Tr{aarg.mapper_train_frac:.3f}'
        if d.startswith(pattern):
            mapper_path = os.path.join(mapper_path, d) + '/'
            matches.append(mapper_path)
    assert len(matches) > 0
    mapper_path_1 = None
    for attacker_path in matches:
        mapper_path_1 = attacker_path + f'{aarg.mapper_targets}/'
        l = sorted(list(os.listdir(mapper_path_1)), key=lambda x: float(x.split('_')[-1]))[0]
        mapper_path_1 = os.path.join(mapper_path_1, l)
        if not os.path.exists(mapper_path_1):
            mapper_path_1 = None
    if mapper_path_1:
        return LMMapper.from_pretrained(mapper_path_1)
    return None


class EmbeddingInversionAttacker(Attacker):
    arg_clz = EIAArguments

    def __init__(self):
        super().__init__()
        self.mapper: LMMapper | None = None
        self.pk = None

    def parse_arguments(self, args, prefix: str):
        res: EIAArguments = super().parse_arguments(args, prefix)
        if res.mapper_dataset is None or len(res.mapper_dataset) == 0:
            res.mapper_dataset = args.dataset
        if ',' in res.mapper_dataset:
            res.train_label = DRA_train_label[res.mapper_dataset.split(',')[0]]
        else:
            res.train_label = DRA_train_label[res.mapper_dataset]
        if res.mapper_target_model_load_bits < 0:
            res.mapper_target_model_load_bits = args.load_bits
        if res.mapper_target_model_name is None or len(res.mapper_target_model_name) == 0:
            res.mapper_target_model_name = args.model_name
        return res

    def load_attacker(self, args, aargs: EIAArguments, llm: SplitWrapperModel = None, tokenizer: Tokenizer = None):
        if aargs.mapper_targets is not None and len(aargs.mapper_targets) > 0:
            self.mapper = get_eia_mapper(aargs)

    def attack(self, args, aargs: EIAArguments, llm: SplitWrapperModel, tokenizer: Tokenizer,
               simulator: SFLSimulator, batch, b2tr_inter, tr2t_inter, all_inters, init=None) -> \
            dict[str, Any]:
        if self.mapper:
            self.mapper.to(llm.device)
        inter = b2tr_inter
        if aargs.at == 'tr2t':
            inter = tr2t_inter
        pk = simulator.parameter_keeper
        if self.pk:
            pk = self.pk
        with ParamRestored(llm=llm, param_keeper=pk, key='pretrained',
                           parts=['bottom']):
            with FLConfigHolder(llm) as ch:
                llm.fl_config.attack_mode = aargs.at
                llm.fl_config.collect_intermediates = False
                llm.fl_config.noise_mode = 'none'
                if self.mapper and aargs.mapper_targets is not None:
                    llm.fl_config.split_point_1 = int(aargs.mapper_targets.split('-')[1])
                ch.change_config()
                if init is not None:
                    dummy = init.clone().detach().to(llm.device)  # (batch_size, seq_len, vocab_size)
                else:
                    dummy = torch.rand((inter.fx.shape[0], inter.fx.shape[1], llm.config.vocab_size)).to(llm.device)
                    # if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                    #     dummy = dummy.permute(1, 0, 2)
                # dummy = torch.softmax(dummy / temp, -1)  # (batch_size, seq_len, vocab_size)
                pbar = tqdm(total=aargs.epochs)
                avg_rglf = 0
                avg_step = 0
                target = inter.fx.to(llm.device).float()
                if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                    target = target.permute(1, 0, 2)
                if self.mapper:
                    target = self.mapper(target).detach()
                dummy.requires_grad = True
                opt = torch.optim.AdamW([dummy], lr=aargs.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=aargs.wd)
                embedding_matrix = get_embedding_matrix(llm).float()  # (vocab, embed_size)
                for e in range(aargs.epochs):
                    opt.zero_grad()
                    dmy = torch.softmax(dummy.float() / aargs.temp, -1) @ embedding_matrix
                    if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                        dmy = torch.softmax(dummy / aargs.temp, -1) @ embedding_matrix
                        it = llm(inputs_embeds=dmy)
                        it = it.permute(1, 0, 2).float()
                    else:
                        it = llm(inputs_embeds=dmy)
                    loss = 0
                    for x, y in zip(it, target):
                        loss += ((x - y) ** 2).sum()
                    # + 0.1 * torch.abs(x - y).sum().float()
                    loss.backward()
                    opt.step()
                    if e % 10 == 0:
                        rg, _, _ = evaluate_attacker_rouge(tokenizer, dummy, batch)
                        avg_rglf += rg["rouge-l"]["f"]
                        avg_step += 1

                    pbar.set_description(
                        f'Epoch {e}/{aargs.epochs} Loss: {loss.item():.5f} ROUGE: {0 if avg_step == 0 else avg_rglf / avg_step :.5f}')
                    pbar.update(1)
        return dummy
