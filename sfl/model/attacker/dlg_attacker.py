from abc import ABC
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import tqdm
from tokenizers import Tokenizer
from torch import nn
from torch.backends.cuda import SDPBackend
from torch.nn import CrossEntropyLoss
from torch.nn import ModuleList
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

from sfl.utils.args import FLConfig
from sfl.model.attacker.base import Attacker
from sfl.model.llm.falcon.falcon_wrapper import FalconForCausalLMSplit
from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.gpt2.gpt2_wrapper import GPT2SplitLMHeadModel
from sfl.model.llm.gptj.gptj_wrapper import GPTJForCausalLMSplit
from sfl.model.llm.llama2.llama2_wrapper import LLAMA2SplitLMHeadModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.t5.t5wrapper import T5ForConditionalGenerationSplitModel
from sfl.simulator.simulator import SFLSimulator, ParamRestored
from sfl.utils.exp import load_model_in_param_keepers
from sfl.utils.model import calc_shifted_loss_logits, get_output


class DLGAttacker(Attacker, ABC):
    """
    Base Class for Gradient-matching-based Attackers
    """

    def __init__(self, fl_config: FLConfig, model: SplitWrapperModel):
        super().__init__()
        self.top_mocker = None
        self.target_model_config = model.config
        self.fl_config = fl_config
        self.model_type = model.type

    def load_attacker(self, args, aargs, llm: SplitWrapperModel = None, tokenizer: Tokenizer = None):
        # this cannot be removed
        print(get_output("test", tokenizer, llm))
        # Load the mimic Top Model
        mocker = None
        assert llm.fl_config is not None
        if isinstance(llm, GPT2SplitLMHeadModel):
            mocker = GPT2TopMocker(llm.fl_config, llm)
        elif isinstance(llm, LLAMA2SplitLMHeadModel):
            mocker = LLAMA2TopMocker(llm.fl_config, llm)
            # print(llm.config)
        elif isinstance(llm, T5ForConditionalGenerationSplitModel):
            mocker = T5DecoderTopMocker(llm.fl_config, llm)
        elif isinstance(llm, ChatGLMForConditionalGenerationSplit):
            mocker = ChatGLMTopMocker(llm.fl_config, llm)
        elif isinstance(llm, GPTJForCausalLMSplit):
            mocker = GPTJTopMocker(llm.fl_config, llm)
        elif isinstance(llm, FalconForCausalLMSplit):
            mocker = FalconTopMocker(llm.fl_config, llm)
        self.top_mocker = mocker

    def _gma_loss(self, inter, gradient, gt, beta=0.85, ppl=False, llm=None, **kwargs):
        self.top_mocker.to(llm.device)
        x = self.top_mocker(inter, **kwargs)
        if self.model_type == 'encoder-decoder':
            loss = CrossEntropyLoss(ignore_index=-100)(x.view(-1, x.size(-1)),
                                                       torch.softmax(gt, dim=-1).view(-1, gt.size(-1)))
        else:
            loss = calc_shifted_loss_logits(x, torch.softmax(gt, dim=-1))
        grad = torch.autograd.grad(loss, inter, create_graph=True)
        grad_diff = 0
        # TAG gradient-matching loss
        for gx, gy in zip(grad[0], gradient.to(loss.device)):
            grad_diff += beta * ((gx - gy) ** 2).sum() + (1 - beta) * (torch.abs(gx - gy)).sum()
        if ppl:
            with torch.no_grad():
                input_ids = gt.argmax(-1)
                ppl = llm(input_ids=input_ids, labels=input_ids).loss
                grad_diff += 0.2 * ppl
        return grad_diff


@dataclass
class TAGArguments:
    enable: bool = False
    epochs: int = 300
    beta: float = 0.85
    lr: float = 0.09
    init_temp: float = 1.0
    softmax: bool = True
    cross_model: str = None


def _extract_args_from_inters(all_inter):
    atk_args = {k[4:]: v.fx for k, v in all_inter.items() if isinstance(k, str) and k.startswith('atk_')}
    return atk_args


class TAGAttacker(DLGAttacker):
    """
    TAG Gradient Attacker, used for TAG, BiSR and BiSR(b)
    """
    arg_clz = TAGArguments

    def __init__(self, fl_config: FLConfig, model: SplitWrapperModel):
        super().__init__(fl_config, model)
        self.pk = None

    def load_attacker(self, args, aargs, llm: SplitWrapperModel = None, tokenizer: Tokenizer = None):
        super().load_attacker(args, aargs, llm, tokenizer)
        if aargs.cross_model is not None and len(aargs.cross_model) > 0 and aargs.cross_model != 'none':
            self.pk = load_model_in_param_keepers(aargs.cross_model, llm.fl_config, ['top'])

    def attack(self, args, aargs: arg_clz,
               llm: SplitWrapperModel, tokenizer: Tokenizer, simulator: SFLSimulator, batch,
               b2tr_inter, tr2t_inter, all_inter, init=None):
        atk_args = _extract_args_from_inters(all_inter)
        gradient = tr2t_inter.grad.clone().to(llm.device)
        inter = tr2t_inter.fx.clone().to(llm.device)
        pk = simulator.parameter_keeper
        if self.pk is not None:
            pk = self.pk
        with ParamRestored(llm=llm, param_keeper=pk, key='pretrained',
                           parts=['top'],
                           write_back=False):
            if args.model_name == 'chatglm':
                gradient = gradient.permute(1, 0, 2)
                inter = inter.permute(1, 0, 2)
            batch_size, seq_len = inter.shape[:2]
            vocab_size = self.target_model_config.vocab_size
            if init is not None:
                if init.shape[1] != gradient.shape[1]:
                    init = init[:, -gradient.shape[1]:, :]
                dra_attacked = init.clone().detach().to(inter.device)  # (batch_size, seq_len, vocab_size)
                # softmax with temperature
                if aargs.softmax:
                    dra_attacked = torch.softmax(dra_attacked / aargs.init_temp, dim=-1)
                gt = dra_attacked.clone()
            else:
                gt = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(inter.device), dim=-1)
            gt.requires_grad = True
            inter.requires_grad = True
            optimizer = torch.optim.AdamW([gt], lr=aargs.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
            for i in range(aargs.epochs):
                optimizer.zero_grad()
                grad_diff = self._gma_loss(inter, gradient, gt, beta=aargs.beta, llm=llm, **atk_args)
                grad_diff.backward()
                optimizer.step()
        return gt


class LAMPArguments(TAGArguments):
    lamp_freq = 30


class LAMPAttacker(DLGAttacker):
    """
    LAMP Attacker, used for LAMP
    """
    arg_clz = LAMPArguments

    def __init__(self, fl_config: FLConfig, model: SplitWrapperModel):
        super().__init__(fl_config, model)

    def _lamp(self, llm, inter, gradient, gt, beta=0.85, lamp_steps=100, **kwargs):
        """
        Copied from the original implementation
        """
        inter.requires_grad = True
        best_gt, best_loss = None, None
        init_loss = None
        changed = None
        batch_size, seq_len, vocab_size = gt.shape
        pbar = tqdm.tqdm(total=lamp_steps)
        for sample_idx in range(lamp_steps):
            new_gt = gt.clone()
            with torch.no_grad():
                for sen_id in range(batch_size):
                    if sample_idx != 0:
                        perm_ids = np.arange(seq_len)
                        if sample_idx != 0:
                            if sample_idx % 4 == 0:  # swap two tokens
                                i, j = 1 + np.random.randint(seq_len - 2), 1 + np.random.randint(seq_len - 2)
                                perm_ids[i], perm_ids[j] = perm_ids[j], perm_ids[i]
                            elif sample_idx % 4 == 1:  # move a token to another place
                                i = 1 + np.random.randint(seq_len - 2)
                                j = 1 + np.random.randint(seq_len - 1)
                                if i < j:
                                    perm_ids = np.concatenate(
                                        [perm_ids[:i], perm_ids[i + 1:j], perm_ids[i:i + 1], perm_ids[j:]])
                                else:
                                    perm_ids = np.concatenate(
                                        [perm_ids[:j], perm_ids[i:i + 1], perm_ids[j:i], perm_ids[i + 1:]])
                            elif sample_idx % 4 == 2:  # move a sequence to another place
                                b = 1 + np.random.randint(seq_len - 1)
                                e = 1 + np.random.randint(seq_len - 1)
                                if b > e:
                                    b, e = e, b
                                p = 1 + np.random.randint(seq_len - 1 - (e - b))
                                if p >= b:
                                    p += e - b
                                if p < b:
                                    perm_ids = np.concatenate(
                                        [perm_ids[:p], perm_ids[b:e], perm_ids[p:b], perm_ids[e:]])
                                elif p >= e:
                                    perm_ids = np.concatenate(
                                        [perm_ids[:b], perm_ids[e:p], perm_ids[b:e], perm_ids[p:]])
                                else:
                                    assert False
                            elif sample_idx % 4 == 3:  # take some prefix and put it at the end
                                i = 1 + np.random.randint(seq_len - 2)
                                perm_ids = np.concatenate(
                                    [perm_ids[:1], perm_ids[i:-1], perm_ids[1:i], perm_ids[-1:]])
                        new_gt[sen_id] = gt[sen_id, perm_ids, :]

            new_gt.requires_grad = True
            loss = self._gma_loss(inter, gradient, new_gt, ppl=True, llm=llm, beta=beta, **kwargs).detach().cpu().item()
            if sample_idx == 0:
                init_loss = loss
            pbar.set_postfix({'loss': loss, 'best_loss': best_loss, 'init_loss': init_loss})
            if (best_loss is None) or (loss < best_loss):
                best_gt = new_gt
                best_loss = loss
                if sample_idx != 0:
                    changed = sample_idx % 4
            pbar.update()
        if not (changed is None):
            change = ['Swapped tokens', 'Moved token', 'Moved sequence', 'Put prefix at the end'][changed]
            print(change)
        return best_gt

    def attack(self, args, aargs: arg_clz,
               llm: SplitWrapperModel, tokenizer: Tokenizer, simulator: SFLSimulator, batch,
               b2tr_inter, tr2t_inter, all_inter, init=None):
        atk_args = _extract_args_from_inters(all_inter)
        gradient = tr2t_inter.grad.clone().to(llm.device)
        inter = tr2t_inter.fx.clone().to(llm.device)
        with ParamRestored(llm=llm, param_keeper=simulator.parameter_keeper, key='pretrained',
                           parts=['bottom', 'top', 'trunk'],
                           write_back=False):
            if args.model_name == 'chatglm':
                gradient = gradient.permute(1, 0, 2)
                inter = inter.permute(1, 0, 2)
            batch_size, seq_len = inter.shape[:2]
            vocab_size = self.target_model_config.vocab_size
            if init is not None:
                if init.shape[1] != gradient.shape[1]:
                    init = init[:, -gradient.shape[1]:, :]
                dra_attacked = init.clone().detach().to(inter.device)  # (batch_size, seq_len, vocab_size)
                # softmax with temperature
                if aargs.softmax:
                    dra_attacked = torch.softmax(dra_attacked / aargs.init_temp, dim=-1)
                gt = dra_attacked.clone()
            else:
                gt = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(inter.device), dim=-1)
            gt.requires_grad = True
            inter.requires_grad = True
            optimizer = torch.optim.AdamW([gt], lr=aargs.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
            for i in range(aargs.epochs):
                optimizer.zero_grad()
                grad_diff = self._gma_loss(inter, gradient, gt, llm=llm, beta=aargs.beta, **atk_args)
                grad_diff.backward()
                optimizer.step()
                if i % aargs.lamp_freq == 0:
                    new_gt = self._lamp(llm, inter, gradient, gt.detach().clone(), beta=aargs.beta, **atk_args)
                    with torch.no_grad():
                        # copy new_gt's data to gt
                        gt.data = new_gt.data
        return gt


class TopMocker(nn.Module):
    """
    Mimic the Top part of LLM
    """
    def __init__(self, fl_config: FLConfig, llm: SplitWrapperModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fl_config = fl_config
        self.model_type = llm.type


class GPT2TopMocker(TopMocker):

    def __init__(self, fl_config: FLConfig, model: GPT2SplitLMHeadModel, **kwargs):
        super().__init__(fl_config, model, **kwargs)
        num_blocks = model.config.n_layer - fl_config.split_point_2
        model.config_sfl(fl_config, None)
        self.decoder = ModuleList([GPT2Block(model.config, i) for i in range(num_blocks)])
        self.ln = nn.LayerNorm(model.transformer.embed_dim, eps=model.config.layer_norm_epsilon)
        # copy the parameters
        for (nm1, param1), (nm2, param2) in zip(model.get_top_params(), self.named_parameters()):
            param2.data = param1.data.clone()
        self.head = deepcopy(model.lm_head)

    def forward(self, x, **kwargs):
        for block in self.decoder:
            x = block(x)[0]
        x = self.ln(x)
        return self.head(x)


class GPTJTopMocker(TopMocker):

    def __init__(self, fl_config: FLConfig, model: GPTJForCausalLMSplit, **kwargs):
        super().__init__(fl_config, model, **kwargs)
        num_blocks = model.config.n_layer - fl_config.split_point_2
        model.config_sfl(fl_config)
        self.decoder = deepcopy(model.transformer.h[-num_blocks:])
        self.ln = deepcopy(model.transformer.ln_f)
        self.head = deepcopy(model.lm_head)

    def forward(self, x, **kwargs):
        for block in self.decoder:
            x = block(x)[0]
        x = self.ln(x)
        return self.head(x)


class FalconTopMocker(TopMocker):
    def __init__(self, fl_config: FLConfig, model: FalconForCausalLMSplit, **kwargs):
        super().__init__(fl_config, model, **kwargs)
        num_blocks = model.config.num_hidden_layers - fl_config.split_point_2
        model.config_sfl(fl_config)
        self.layers = model.transformer.h[-num_blocks:]
        self.ln = model.transformer.ln_f
        self.head = model.lm_head

    def forward(self, x, attention_mask, alibi, **kwargs):
        if attention_mask is not None:
            attention_mask = attention_mask.to(x.device)
        if alibi is not None:
            alibi = alibi.to(x.device)
        for block in self.layers:
            x = block(x, attention_mask=attention_mask, alibi=alibi)[0]
        x = self.ln(x)
        return self.head(x)


class LLAMA2TopMocker(TopMocker):

    def __init__(self, fl_config: FLConfig, model: LLAMA2SplitLMHeadModel, **kwargs):
        super().__init__(fl_config, model, **kwargs)
        num_blocks = model.config.num_hidden_layers - fl_config.split_point_2
        model.config_sfl(fl_config, None)
        self.layers = model.model.layers[-num_blocks:]
        self.ln = model.model.norm
        self.head = model.lm_head

    def forward(self, x, **kwargs):
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + x.shape[1], device=x.device
        )

        position_ids = cache_position.unsqueeze(0)
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            for block in self.layers:
                x = block(x, position_ids=position_ids)[0]
        x = self.ln(x)
        return self.head(x)


class T5DecoderTopMocker(TopMocker):

    def __init__(self, fl_config: FLConfig, model: T5ForConditionalGenerationSplitModel, **kwargs):
        super().__init__(fl_config, model, **kwargs)
        num_blocks = model.config.num_decoder_layers - fl_config.split_point_2
        self.blocks = nn.ModuleList(
            [T5Block(model.decoder.config, has_relative_attention_bias=bool(i == 0 and fl_config.split_point_2 == 0))
             for i in range(num_blocks)]
        )
        self.final_layer_norm = T5LayerNorm(model.decoder.config.d_model, eps=model.decoder.config.layer_norm_epsilon)
        # copy the parameters
        for (nm1, param1), (nm2, param2) in zip(model.get_top_params(), self.named_parameters()):
            param2.data = param1.data.clone()
        self.lm_head = deepcopy(model.lm_head)

    def forward(self, x, encoder_inter, **kwargs):
        for block in self.blocks:
            x = block(x, encoder_hidden_states=encoder_inter)[0]
        x = self.final_layer_norm(x)
        return self.lm_head(x)


class ChatGLMTopMocker(TopMocker):

    def __init__(self, fl_config: FLConfig, model: ChatGLMForConditionalGenerationSplit, *args, **kwargs):
        super().__init__(fl_config, model, *args, **kwargs)
        num_blocks = model.config.num_layers - fl_config.split_point_2
        self.blocks = deepcopy(model.transformer.encoder.layers[-num_blocks:])
        self.final_ln = deepcopy(model.transformer.encoder.final_layernorm)
        self.output_layer = deepcopy(model.transformer.output_layer)

    def forward(self, x, attention_mask, rotary_pos_emb, **kwargs):
        if attention_mask is not None:
            attention_mask = attention_mask.to(x.device)
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb.to(x.device)
        x = x.permute(1, 0, 2)
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            for block in self.blocks:
                x = block(x, attention_mask=attention_mask,
                          rotary_pos_emb=rotary_pos_emb)[0]
        x = self.final_ln(x)
        return self.output_layer(x).permute(1, 0, 2)
