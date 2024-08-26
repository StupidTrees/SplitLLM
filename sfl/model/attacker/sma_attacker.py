from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from sfl.model.attacker.base import Attacker
from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator, ParamRestored
from sfl.utils.exp import load_model_in_param_keepers
from sfl.utils.model import get_embedding_layer, FLConfigHolder, evaluate_attacker_rouge


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
    """
    Smashed Data Matching Attacker, User for BiSR and BiSR (f)
    """
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

        if aargs.at == 'b2tr':
            inter = b2tr_inter
        elif aargs.at == 'tr2t':
            inter = tr2t_inter
        else:
            inter = all_inters[int(aargs.at)]
        pk = simulator.parameter_keeper
        if self.pk:
            pk = self.pk
        with ParamRestored(llm=llm, param_keeper=pk, key='pretrained',
                           parts=['bottom']):
            with FLConfigHolder(llm) as ch:
                if aargs.at in ['b2tr', 'tr2t']:
                    llm.fl_config.attack_mode = aargs.at
                else:
                    llm.fl_config.split_point_1 = int(aargs.at)
                    llm.fl_config.attack_mode = 'b2tr'
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
                    if e % 10 == 0:
                        texts = self._rec_text(llm, dummy)
                        rg, _, _ = evaluate_attacker_rouge(tokenizer, texts, batch)
                        avg_rglf += rg["rouge-l"]["f"]
                        avg_step += 1
                    pbar.set_description(
                        f'Epoch {e}/{aargs.epochs} Loss: {loss.item()} ROUGE: {0 if avg_step == 0 else avg_rglf / avg_step}')
                    pbar.update(1)
        return self._rec_text(llm, dummy)
