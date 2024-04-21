from copy import deepcopy

import torch
from tqdm import tqdm

from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.model import Intermediate, evaluate_attacker_rouge, get_embedding_layer


class WhiteBoxMIAttacker(object):

    def fit(self, tok, llm: SplitWrapperModel, b2tr_inter: Intermediate, gt, epochs=15000, lr=1e-3, dummy_init=None):
        # generate random tensor same size as gt
        if dummy_init is not None:
            dummy = dummy_init.clone().detach().to(llm.device).argmax(-1)
        else:
            dummy = torch.randint(0, llm.config.vocab_size, b2tr_inter.fx.shape[:-1]).to(llm.device)
            # convert to long
            dummy = dummy.long()
        dummy = get_embedding_layer(llm)(dummy)

        cfg_bk = deepcopy(llm.fl_config)
        fl_config = llm.fl_config
        fl_config.attack_mode = 'b2tr'
        fl_config.collect_intermediates = False
        llm.config_sfl(fl_config, None)
        pbar = tqdm(total=epochs)
        avg_rglf = 0
        avg_step = 0
        dummy.requires_grad = True
        opt = torch.optim.Adam([dummy], lr=lr, eps=1e-8, weight_decay=0.001)
        for e in range(epochs):
            opt.zero_grad()
            inter = llm(inputs_embeds=dummy)
            # loss = torch.nn.CosineSimilarity()
            loss = torch.nn.MSELoss()
            loss = loss(inter, b2tr_inter.fx.to(llm.device)).mean()
            loss.backward()
            opt.step()
            pbar.update(1)
            if e % 10 == 0:
                texts = self.rec_text(llm, dummy)
                rg, _, _ = evaluate_attacker_rouge(tok, texts, gt)
                avg_rglf += rg["rouge-l"]["f"]
                avg_step += 1
            logits = self.rec_text(llm, dummy)
            text = tok.decode(logits.argmax(-1)[0], skip_special_tokens=True)
            pbar.set_description(
                f'Epoch {e}/{epochs} Loss: {loss.item()} ROUGE: {0 if avg_step == 0 else avg_rglf / avg_step}, Text:{text}')
            # dummy = self.rec_text(llm, dummy).argmax(-1)
            # dummy = get_embedding_layer(llm)(dummy)
        llm.config_sfl(cfg_bk, None)
        return self.rec_text(llm, dummy)

    def rec_text(self, llm, embeds):
        wte = get_embedding_layer(llm)
        all_words = torch.tensor(list([i for i in range(llm.config.vocab_size)])).to(llm.device)
        all_embeds = wte(all_words)
        cosine_similarities = torch.matmul(embeds, all_embeds.transpose(0, 1))  # (bs, seq,vocab)
        return cosine_similarities
