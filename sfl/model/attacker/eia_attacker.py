import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import Linear
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel

from sfl.config import MapperConfig, DRA_train_label
from sfl.model.attacker.attacker import Attacker
from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.param_keeper import InMemoryParameterKeeper
from sfl.simulator.simulator import SFLSimulator, ParamRestored
from sfl.utils.exp import get_model_and_tokenizer, required_quantization, load_model_in_param_keepers
from sfl.utils.model import evaluate_attacker_rouge, get_embedding_layer, FLConfigHolder, \
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


def get_mapper_config(args) -> MapperConfig:
    res = MapperConfig()
    res.path = args.mapper_path
    res.dataset = args.mapper_dataset
    if args.mapper_dataset is None or len(args.mapper_dataset) == 0:
        res.dataset = args.dataset
    if ',' in res.dataset:
        res.train_label = DRA_train_label[res.dataset.split(',')[0]]
    else:
        res.train_label = DRA_train_label[res.dataset]
    res.target_model_load_bits = args.load_bits
    # if res.target_model_load_bits < 0:
    #     res.target_model_load_bits = args.load_bits
    res.train_frac = args.mapper_train_frac
    if args.mapper_target is None or len(args.mapper_target) == 0:
        args.mapper_target = f'{args.split_points.split("-")[0]}-1'
    res.from_layer = int(args.mapper_target.split('-')[0])
    res.to_layer = int(args.mapper_target.split('-')[1])
    res.target_dataset = args.dataset
    res.target_model_name = args.model_name
    return res


def get_eia_mapper(mapper_config: MapperConfig):
    dataset = mapper_config.dataset
    if dataset is None:
        dataset = mapper_config.target_dataset

    model_name = mapper_config.target_model_name
    if required_quantization(mapper_config.target_model_name):
        model_name += f"-{mapper_config.target_model_load_bits}bits"
    mapper_path = mapper_config.path + f'{model_name}/{dataset}/'
    matches = []
    for d in os.listdir(mapper_path):
        pattern = f'{DRA_train_label[dataset]}*{mapper_config.train_frac:.3f}'
        if ',' in mapper_config.dataset:
            pattern = f'Tr{mapper_config.train_frac:.3f}'
        if d.startswith(pattern):
            mapper_path = os.path.join(mapper_path, d) + '/'
            matches.append(mapper_path)
    assert len(matches) > 0
    mapper_path_1 = None
    for attacker_path in matches:
        mapper_path_1 = attacker_path + f'{mapper_config.from_layer}-{mapper_config.to_layer}/'
        l = sorted(list(os.listdir(mapper_path_1)), key=lambda x: float(x.split('_')[-1]))[
            -1 if mapper_config.larger_better else 0]
        mapper_path_1 = os.path.join(mapper_path_1, l)
        if not os.path.exists(mapper_path_1):
            mapper_path_1 = None
    if mapper_path_1:
        return LMMapper.from_pretrained(mapper_path_1)
    return None


@dataclass
class EIAArguments:
    enable: bool = False
    at: str = 'b2tr'
    mapped_to: int = 1
    epochs: int = 72000
    lr: float = 0.09
    wd: float = 0.01
    temp: float = 0.2
    cross_model: str = None


class EmbeddingInversionAttacker(Attacker):
    arg_clz = EIAArguments

    def __init__(self):
        super().__init__()
        self.mapper: LMMapper | None = None
        self.pk = None

    def load_attacker(self, args, aargs: EIAArguments, llm: SplitWrapperModel = None, tokenizer: Tokenizer = None):
        mapper_config = get_mapper_config(args)
        if aargs.cross_model is not None and len(aargs.cross_model) > 0 and aargs.cross_model != 'none':
            mapper_config.target_model_name = aargs.cross_model
            self.pk = load_model_in_param_keepers(aargs.cross_model, llm.fl_config, ['bottom'])
        if aargs.mapped_to > 0:
            self.mapper = get_eia_mapper(mapper_config)

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
                if self.mapper and aargs.mapped_to > 0:
                    llm.fl_config.split_point_1 = aargs.mapped_to
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


@dataclass
class SMArguments:
    enable: bool = False
    at: str = 'b2tr'
    epochs: int = 20
    lr: float = 1e-4
    wd: float = 0.01
    cosine_loss: bool = True
    cross_model: str = None
    init: bool = True


class SmashedDataMatchingAttacker(Attacker):
    arg_clz = SMArguments

    def load_attacker(self, args, aargs: arg_clz, llm: SplitWrapperModel = None, tokenizer: Tokenizer = None):
        self.pk = None
        if aargs.cross_model is not None and len(aargs.cross_model) > 0 and aargs.cross_model != 'none':
            self.pk = load_model_in_param_keepers(aargs.cross_model, llm.fl_config, ['bottom'])

    def _rec_text(self, llm, embeds):
        wte = get_embedding_layer(llm)
        all_words = torch.tensor(list([i for i in range(llm.config.vocab_size)])).to(llm.device)
        all_embeds = wte(all_words)
        if isinstance(llm, ChatGLMForConditionalGenerationSplit):
            embeds = embeds.permute(1, 0, 2)
            all_embeds = all_embeds.permute(1, 0)
        if all_embeds.dtype == torch.float16:
            embeds = embeds.float()
            all_embeds = all_embeds.float()
        cosine_similarities = torch.matmul(embeds, all_embeds.transpose(0, 1))  # (bs, seq,vocab)
        return torch.softmax(cosine_similarities, -1)

    def attack(self, args, aargs: SMArguments, llm: SplitWrapperModel, tokenizer: Tokenizer,
               simulator: SFLSimulator, batch, b2tr_inter, tr2t_inter, all_inters, init=None):
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
                ch.change_config()

                if init is not None and aargs.init:
                    dummy = init.clone().detach().to(llm.device).argmax(-1)
                else:
                    dummy = torch.randint(0, llm.config.vocab_size, inter.fx.shape[:-1]).to(llm.device)
                    dummy = dummy.long()
                    if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                        dummy = dummy.permute(1, 0)

                dummy = get_embedding_layer(llm)(dummy)
                if dummy.dtype == torch.float16:
                    dummy = dummy.float()

                pbar = tqdm(total=aargs.epochs)
                avg_rglf = 0
                avg_step = 0
                dummy.requires_grad = True
                opt = torch.optim.AdamW([dummy], lr=aargs.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=aargs.wd)
                for e in range(aargs.epochs):
                    opt.zero_grad()
                    dmy = dummy
                    # if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                    #     dmy = dummy.half()
                    it = llm(inputs_embeds=dmy)
                    target = inter.fx.to(llm.device)
                    if it.dtype == torch.float16:
                        it = it.float()
                    if target.dtype == torch.float16:
                        target = target.float()
                    if dummy.dtype == torch.float16:
                        dummy = dummy.float()
                    if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                        target = target.permute(1, 0, 2).contiguous()
                        it = it.permute(1, 0, 2).contiguous()
                    loss = 0
                    if aargs.cosine_loss:
                        for x, y in zip(it, target):
                            loss += 1 - torch.cosine_similarity(x, y, dim=-1).mean()
                    else:
                        for x, y in zip(it, target):
                            loss += ((x - y) ** 2).mean()  # + 0.1 * torch.abs(x - y).sum().float()

                    loss.backward()
                    opt.step()
                    # print(f"Loss:{loss.item()} Before: {sent_before} After: {sent_after}")

                    if e % 10 == 0:
                        texts = self._rec_text(llm, dummy)
                        rg, _, _ = evaluate_attacker_rouge(tokenizer, texts, batch)
                        avg_rglf += rg["rouge-l"]["f"]
                        avg_step += 1
                    pbar.set_description(
                        f'Epoch {e}/{aargs.epochs} Loss: {loss.item()} ROUGE: {0 if avg_step == 0 else avg_rglf / avg_step}')
                    pbar.update(1)
        return self._rec_text(llm, dummy)
