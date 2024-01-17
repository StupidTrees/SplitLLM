from copy import deepcopy

import torch
from torch import nn
from torch.nn import ModuleList
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from sfl.config import FLConfig
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.model.split_model import SplitModel
from sfl.utils.training import calc_shifted_loss_logits


class Mocker(nn.Module):
    def __init__(self, fl_config: FLConfig, model: SplitModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_model_config = model.config
        self.fl_config = fl_config

    def fit(self, inter, gradient, epochs=300, adjust=0, beta=0.5, lr=0.09):
        batch_size, seq_len = inter.shape[:2]
        vocab_size = self.target_model_config.vocab_size
        gt = torch.softmax(torch.randn((batch_size, seq_len, vocab_size)).to(inter.device), dim=-1)
        gt.requires_grad = True
        inter.requires_grad = True
        optimizer = torch.optim.AdamW([gt], lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        for i in range(epochs):
            optimizer.zero_grad()
            x = self.forward(inter)
            loss = calc_shifted_loss_logits(x, torch.softmax(gt, dim=-1))
            grad = torch.autograd.grad(loss, inter, create_graph=True)
            grad_diff = 0
            for gx, gy in zip(grad, gradient.to(loss.device)):
                grad_diff += beta * ((gx - gy) ** 2).sum() + (1 - beta) * (torch.abs(gx - gy)).sum()
            grad_diff.backward()
            optimizer.step()
        if adjust > 0:
            optimizer2 = torch.optim.AdamW(self.parameters(), lr=1e-4)
            for i in range(adjust):
                optimizer2.zero_grad()
                x = self.forward(inter)
                loss = calc_shifted_loss_logits(x, torch.softmax(gt, dim=-1))
                grad = torch.autograd.grad(loss, inter, create_graph=True)
                grad_diff = 0
                for gx, gy in zip(grad, gradient.to(loss.device)):
                    grad_diff += beta * ((gx - gy) ** 2).sum() + (1 - beta) * (torch.abs(gx - gy)).sum()
                loss = loss + 0.5 * grad_diff
                loss.backward()
                optimizer2.step()
        return gt


class GPT2TopMocker(Mocker):

    def __init__(self, fl_config: FLConfig, model: GPT2SplitLMHeadModel):
        super().__init__(fl_config, model)
        num_blocks = model.config.n_layer - fl_config.split_point_2
        self.decoder = ModuleList([GPT2Block(model.config, i) for i in range(num_blocks)])
        self.ln = nn.LayerNorm(model.transformer.embed_dim, eps=model.config.layer_norm_epsilon)
        # copy the parameters
        for (nm1, param1), (nm2, param2) in zip(model.get_top_params(), self.named_parameters()):
            param2.data = param1.data.clone()
        self.head = deepcopy(model.lm_head)

    def forward(self, x):
        for block in self.decoder:
            x = block(x)[0]
        x = self.ln(x)
        return self.head(x)
