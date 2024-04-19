from copy import deepcopy

import torch
from torch import float16

from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.gpt2.gpt2_wrapper import GPT2SplitLMHeadModel
from sfl.model.llm.llama2.llama2_wrapper import LLAMA2SplitLMHeadModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.model import Intermediate, evaluate_attacker_rouge
from tqdm import tqdm


class WhiteBoxMIAttacker(object):

    def get_embed(self, llm):
        if isinstance(llm, LLAMA2SplitLMHeadModel):
            wte = llm.model.embed_tokens
        elif isinstance(llm, GPT2SplitLMHeadModel):
            wte = llm.transformer.wte
        elif isinstance(llm, ChatGLMForConditionalGenerationSplit):
            wte = llm.transformer.embedding
        return wte

    def fit(self, tok, llm: SplitWrapperModel, b2tr_inter: Intermediate, gt, epochs=10, lr=1e-12):
        # generate random tensor same size as gt
        dummy = torch.randint(0, llm.config.vocab_size, b2tr_inter.fx.shape[:-1]).to(llm.device)
        # convert to long
        dummy = dummy.long()
        dummy = self.get_embed(llm)(dummy)
        cfg_bk = deepcopy(llm.fl_config)
        fl_config = llm.fl_config
        fl_config.attack_mode = 'b2tr'
        fl_config.collect_intermediates = False
        llm.config_sfl(fl_config, None)
        pbar = tqdm(total=epochs)
        avg_rglf = 0
        avg_step = 0
        dummy.requires_grad = True
        opt = torch.optim.AdamW([dummy], lr=lr)
        for e in range(epochs):
            opt.zero_grad()
            inter = llm(inputs_embeds=dummy)
            cosine_loss = torch.nn.CosineSimilarity()
            loss = 1 - cosine_loss(inter, b2tr_inter.fx.to(llm.device)).mean()
            loss.backward()
            opt.step()
            pbar.update(1)
            if e % 10 == 0:
                texts = self.rec_text(llm, dummy)
                rg, _, _ = evaluate_attacker_rouge(tok, texts, gt)
                avg_rglf += rg["rouge-l"]["f"]
            pbar.set_description(
                f'Epoch {e}/{epochs} Loss: {loss.item()} ROUGE: {0 if avg_step == 0 else avg_rglf / avg_step}')
            dummy = self.rec_text(llm, dummy).argmax(-1)
            dummy = self.get_embed(llm)(dummy)
        llm.config_sfl(cfg_bk, None)
        return self.rec_text(llm, dummy)

    def rec_text(self, llm, embeds):
        wte = self.get_embed(llm)
        all_words = torch.tensor(list([i for i in range(llm.config.vocab_size)])).to(llm.device)
        all_embeds = wte(all_words)
        cosine_similarities = torch.matmul(embeds, all_embeds.transpose(0, 1))  # (bs, seq,vocab)
        return cosine_similarities
