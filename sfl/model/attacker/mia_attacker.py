from copy import deepcopy

import torch
from tqdm import tqdm

from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.model import Intermediate, evaluate_attacker_rouge, get_embedding_layer, FLConfigHolder


class WhiteBoxMIAttacker(object):

    def fit(self, tok, llm: SplitWrapperModel, b2tr_inter: Intermediate, gt, epochs=10, lr=1e-4, dummy_init=None,
            attacker=None, cos_loss=True):
        with FLConfigHolder(llm) as ch:
            llm.fl_config.attack_mode = 'b2tr'
            llm.fl_config.collect_intermediates = False
            ch.change_config()

            if dummy_init is not None:
                dummy = dummy_init.clone().detach().to(llm.device).argmax(-1)
            else:
                dummy = torch.randint(0, llm.config.vocab_size, b2tr_inter.fx.shape[:-1]).to(llm.device)
                dummy = dummy.long()
                if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                    dummy = dummy.permute(1, 0)

            dummy = get_embedding_layer(llm)(dummy)
            if dummy.dtype == torch.float16:
                dummy = dummy.float()

            pbar = tqdm(total=epochs)
            avg_rglf = 0
            avg_step = 0
            dummy.requires_grad = True
            opt = torch.optim.AdamW([dummy], lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
            for e in range(epochs):
                opt.zero_grad()
                inter = llm(inputs_embeds=dummy.half())
                target = b2tr_inter.fx.to(llm.device)
                if inter.dtype == torch.float16:
                    inter = inter.float()
                if target.dtype == torch.float16:
                    target = target.float()
                if dummy.dtype == torch.float16:
                    dummy = dummy.float()
                if attacker is not None:
                    out = attacker(inter)
                    out2 = attacker(target)
                    loss = torch.nn.CrossEntropyLoss()(out, out2)
                else:
                    if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                        target = target.permute(1, 0, 2).contiguous()
                        inter = inter.permute(1, 0, 2).contiguous()
                    loss = 0
                    if cos_loss:
                        for x, y in zip(inter, target):
                            loss += 1 - torch.cosine_similarity(x, y, dim=-1).mean()
                            # loss += ((x - y) ** 2).mean()# + 0.1 * torch.abs(x - y).mean().float()
                    else:
                        for x, y in zip(inter, target):
                            loss += ((x - y) ** 2).mean()  # + 0.1 * torch.abs(x - y).sum().float()

                loss.backward()
                opt.step()
                # print(f"Loss:{loss.item()} Before: {sent_before} After: {sent_after}")

                if e % 10 == 0:
                    texts = self.rec_text(llm, dummy)
                    rg, _, _ = evaluate_attacker_rouge(tok, texts, gt)
                    avg_rglf += rg["rouge-l"]["f"]
                    avg_step += 1
                # logits = self.rec_text(llm, dummy)
                # text = tok.decode(logits.argmax(-1)[0], skip_special_tokens=True)
                pbar.set_description(
                    f'Epoch {e}/{epochs} Loss: {loss.item()} ROUGE: {0 if avg_step == 0 else avg_rglf / avg_step}')
                pbar.update(1)
                # dummy = self.rec_text(llm, dummy).argmax(-1)
                # dummy = get_embedding_layer(llm)(dummy)
        return self.rec_text(llm, dummy)

    def rec_text(self, llm, embeds):
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
        # print(cosine_similarities.shape)
        return torch.softmax(cosine_similarities, -1)
