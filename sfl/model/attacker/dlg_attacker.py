from copy import deepcopy

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import ModuleList
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

from sfl.config import FLConfig
from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.gpt2.gpt2_wrapper import GPT2SplitLMHeadModel
from sfl.model.llm.llama2.llama2_wrapper import LLAMA2SplitLMHeadModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.t5.t5wrapper import T5ForConditionalGenerationSplitModel
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.model import calc_shifted_loss_logits


class DLGAttacker(nn.Module):
    """
    DLG攻击模型
    """

    def __init__(self, fl_config: FLConfig, model: SplitWrapperModel, method='tag', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_model_config = model.config
        self.fl_config = fl_config
        self.model_type = model.type
        self.method = method
        self.simulator: SFLSimulator = None

    def gma_loss(self, inter, gradient, gt, beta=0.85, ppl=False, **kwargs):
        # print(inter.shape, gradient.shape, gt.shape, flush=True)
        x = self.forward(inter, **kwargs)
        if self.model_type == 'encoder-decoder':
            loss = CrossEntropyLoss(ignore_index=-100)(x.view(-1, x.size(-1)),
                                                       torch.softmax(gt, dim=-1).view(-1, gt.size(-1)))
        else:
            loss = calc_shifted_loss_logits(x, torch.softmax(gt, dim=-1))
        grad = torch.autograd.grad(loss, inter, create_graph=True)
        grad_diff = 0
        for gx, gy in zip(grad, gradient.to(loss.device)):
            if self.method == 'dlg':
                grad_diff += ((gx - gy) ** 2).sum()
            else:
                grad_diff += beta * ((gx - gy) ** 2).sum() + (1 - beta) * (torch.abs(gx - gy)).sum()
        if ppl:
            with torch.no_grad():
                input_ids = gt.argmax(-1)
                ppl = self.simulator.llm(input_ids=input_ids, labels=input_ids).loss
                grad_diff += 0.2 * ppl
        return grad_diff

    def fit(self, inter, gradient, epochs=300, adjust=0, beta=0.85, lr=0.09, gt_init=None, init_temp=1.0,
            model_name=None, lamp=False, lamp_freq=30, **kwargs):
        if model_name == 'chatglm':
            gradient = gradient.permute(1, 0, 2)
            inter = inter.permute(1, 0, 2)
        batch_size, seq_len = inter.shape[:2]
        vocab_size = self.target_model_config.vocab_size
        if gt_init is not None:
            if gt_init.shape[1] != gradient.shape[1]:
                gt_init = gt_init[:, -gradient.shape[1]:, :]
            dra_attacked = gt_init.clone().detach().to(inter.device)  # (batch_size, seq_len, vocab_size)
            # softmax with temperature
            dra_attacked = torch.softmax(dra_attacked / init_temp, dim=-1)
            # dra_attacked = torch.softmax(dra_attacked, dim=-1)

            gt = dra_attacked.clone()
        else:
            gt = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(inter.device), dim=-1)
        gt.requires_grad = True
        inter.requires_grad = True
        optimizer = torch.optim.AdamW([gt], lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        for i in range(epochs):
            optimizer.zero_grad()
            grad_diff = self.gma_loss(inter, gradient, gt, beta=beta, **kwargs)
            grad_diff.backward()
            optimizer.step()
            if lamp and i % lamp_freq == 0:
                new_gt = self.lamp(inter, gradient, gt.detach().clone(), beta=beta, **kwargs)
                with torch.no_grad():
                    # copy new_gt's data to gt
                    gt.data = new_gt.data
        if adjust > 0:
            optimizer2 = torch.optim.AdamW(self.parameters(), lr=1e-5)
            for i in range(adjust):
                optimizer2.zero_grad()
                x = self.forward(inter, **kwargs)
                loss = calc_shifted_loss_logits(x, torch.softmax(gt, dim=-1))
                loss.backward()
                optimizer2.step()
        return gt

    def lamp(self, inter, gradient, gt, beta=0.85, lamp_steps=50, **kwargs):
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
                        if self.method == 'bisr':
                            # randomly select 1~seq_len*0.1 tokens
                            num_tokens = np.random.randint(1, int(seq_len * 0.2))
                            # randomly select the positions of the tokens
                            token_indexes = np.random.choice(range(seq_len), num_tokens)
                            # for each position, randomly change the last dimension of the token
                            for i in token_indexes:
                                second_idx = np.random.choice([-1, -2, -3])
                                max_idx = torch.argmax(new_gt[sen_id, i, :])
                                second_max_idx = torch.argsort(new_gt[sen_id, i, :])[second_idx]
                                # swap the two tokens
                                new_gt[sen_id, i, max_idx], new_gt[sen_id, i, second_max_idx] = new_gt[
                                                                                                    sen_id, i, second_max_idx], \
                                                                                                new_gt[
                                                                                                    sen_id, i, max_idx]
                        else:
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
            loss = self.gma_loss(inter, gradient, new_gt, ppl=True, beta=beta, **kwargs).detach().cpu().item()
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


class GPT2TopDLGAttacker(DLGAttacker):

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


class LLAMA2TopDLGAttacker(DLGAttacker):

    def __init__(self, fl_config: FLConfig, model: LLAMA2SplitLMHeadModel, **kwargs):
        super().__init__(fl_config, model, **kwargs)
        num_blocks = model.config.num_hidden_layers - fl_config.split_point_2
        model.config_sfl(fl_config, None)
        self.layers = deepcopy(model.model.layers[-num_blocks:])
        self.ln = deepcopy(model.model.norm)
        self.head = deepcopy(model.lm_head)

    def forward(self, x, **kwargs):
        for block in self.layers:
            x = block(x)[0]
        x = self.ln(x)
        return self.head(x)


class T5DecoderDLGAttacker(DLGAttacker):

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


class ChatGLMDLGAttacker(DLGAttacker):

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
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask,
                      rotary_pos_emb=rotary_pos_emb)[0]
        x = self.final_ln(x)
        return self.output_layer(x).permute(1, 0, 2)
