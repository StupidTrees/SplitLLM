import os
from typing import Any

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from sfl.model.attacker.base import Attacker
from sfl.model.attacker.eia.args import EIAArguments, MapperTrainingArguments
from sfl.model.attacker.eia.mapper_models import LMMapper
from sfl.model.attacker.eia.mapper_training import train_mapper
from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator, ParamRestored
from sfl.utils.args import PrefixArgumentParser
from sfl.utils.exp import required_quantization, get_dra_train_label
from sfl.utils.model import evaluate_attacker_rouge, FLConfigHolder, \
    get_embedding_matrix


def get_eia_mapper(aarg: EIAArguments):
    dataset = aarg.mapper_dataset
    model_name = aarg.mapper_target_model_name
    if aarg.cross_model is not None and len(aarg.cross_model) > 0:
        model_name = aarg.cross_model
    if required_quantization(model_name):
        model_name += f"-{aarg.mapper_target_model_load_bits}bits"
    mapper_path = aarg.mapper_path + f'{model_name}/{dataset}/'
    matches = []
    if not os.path.exists(mapper_path):
        return None
    for d in os.listdir(mapper_path):
        pattern = f'{get_dra_train_label(dataset)}*{aarg.mapper_train_frac:.3f}'
        if ',' in aarg.mapper_dataset:
            pattern = f'Tr{aarg.mapper_train_frac:.3f}'
        if d.startswith(pattern):
            mapper_path = os.path.join(mapper_path, d) + '/'
            matches.append(mapper_path)
    if len(matches) == 0:
        return None
    mapper_path_1 = None
    for attacker_path in matches:
        mapper_path_1 = attacker_path + f'{aarg.mapper_targets}/'
        if not os.path.exists(mapper_path_1):
            mapper_path_1 = None
        else:
            l = sorted(list(os.listdir(mapper_path_1)), key=lambda x: float(x.split('_')[-1]))[0]
            mapper_path_1 = os.path.join(mapper_path_1, l)
            if not os.path.exists(mapper_path_1):
                mapper_path_1 = None
    if mapper_path_1:
        return LMMapper.from_pretrained(mapper_path_1)
    return None


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    return torch.softmax(logits / temperature, dim=-1)
    # y = logits + sample_gumbel(logits.size()).to(logits.device)
    # return torch.softmax(y / temperature, dim=-1)


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
            res.train_label = get_dra_train_label(res.mapper_dataset.split(',')[0])
        else:
            res.train_label = get_dra_train_label(res.mapper_dataset)
        if res.mapper_target_model_load_bits < 0:
            res.mapper_target_model_load_bits = args.load_bits
        if res.mapper_target_model_name is None or len(res.mapper_target_model_name) == 0:
            res.mapper_target_model_name = args.model_name
        return res

    def load_attacker(self, args, aargs: EIAArguments, llm: SplitWrapperModel = None, tokenizer: Tokenizer = None):
        if aargs.mapped_to >= 0 and aargs.mapper_targets is not None and len(aargs.mapper_targets) > 0:
            self.mapper = get_eia_mapper(aargs)
            if not self.mapper:
                print(f'Mapper not found for {aargs.mapper_targets}, start training mapper')
                parser = PrefixArgumentParser(prefix='eia_mapper_training', dataclass_types=[MapperTrainingArguments])
                training_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
                with ParamRestored(llm, llm.param_keeper, ['bottom', 'trunk', 'top'], key='pretrained',
                                   write_back=False, disable_inter_collection=False):
                    with FLConfigHolder(llm):
                        training = llm.training
                        train_mapper(llm, tokenizer, aargs, training_args=training_args)
                        llm.train(training)
                self.mapper = get_eia_mapper(aargs)
                assert self.mapper is not None

    def attack(self, args, aargs: EIAArguments, llm: SplitWrapperModel, tokenizer: Tokenizer,
               simulator: SFLSimulator, batch, b2tr_inter, tr2t_inter, all_inters, init=None) -> \
            dict[str, Any]:
        if self.mapper:
            self.mapper.to(llm.device)
        if aargs.at == 'b2tr':
            inter = b2tr_inter
        elif aargs.at == 'tr2t':
            inter = tr2t_inter
        else:
            inter = all_inters[int(aargs.at) - 1]
        pk = simulator.parameter_keeper
        if self.pk:
            pk = self.pk
        with ParamRestored(llm=llm, param_keeper=pk, key='pretrained',
                           parts=['bottom', 'trunk']):
            with FLConfigHolder(llm) as ch:
                if aargs.at in ['tr2t', 'b2tr']:
                    llm.fl_config.attack_mode = aargs.at
                else:
                    llm.fl_config.attack_mode = 'b2tr'
                llm.fl_config.collect_intermediates = False
                llm.fl_config.noise_mode = 'none'
                if self.mapper and aargs.mapper_targets is not None:
                    llm.fl_config.split_point_1 = int(aargs.mapper_targets.split('-')[1])
                elif aargs.at != 'b2tr' and aargs.at != 'tr2t':
                    llm.fl_config.split_point_1 = int(aargs.at)
                    llm.fl_config.attack_mode = 'b2tr'
                ch.change_config()
                if init is not None:
                    dummy = init.clone().detach().to(llm.device)  # (batch_size, seq_len, vocab_size)
                else:
                    dummy = torch.rand((inter.fx.shape[0], inter.fx.shape[1], llm.config.vocab_size)).to(llm.device)
                    if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                        dummy = dummy.permute(1, 0, 2)
                # dummy = torch.softmax(dummy / temp, -1)  # (batch_size, seq_len, vocab_size)
                pbar = tqdm(total=aargs.epochs)
                rglf = 0
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
                    dmy = gumbel_softmax_sample(dummy.float(), aargs.temp) @ embedding_matrix
                    if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                        dmy = gumbel_softmax_sample(dummy, aargs.temp) @ embedding_matrix
                        dmy = dmy.transpose(1, 0).contiguous()
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
                        rglf = rg["rouge-l"]["f"]

                    pbar.set_description(
                        f'Epoch {e}/{aargs.epochs} Loss: {loss.item():.5f} ROUGE: {rglf:.5f}')
                    pbar.update(1)
        return dummy
