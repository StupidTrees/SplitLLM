from dataclasses import dataclass

import torch
from torch.nn import Linear
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel

from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.model import Intermediate, evaluate_attacker_rouge, get_embedding_layer, FLConfigHolder, \
    get_embedding_matrix


@dataclass
class LMMapperConfig(PretrainedConfig):
    n_embed: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


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
        self.mapper = Linear(config.n_embed, self.config.n_embed)

    def forward(self, hidden_states):
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()
        return self.mapper(hidden_states)


class WhiteBoxMIAttacker(object):

    def fit(self, tok, llm: SplitWrapperModel, inter: Intermediate, gt, epochs=10, lr=1e-4, dummy_init=None,
            attacker=None, cos_loss=True, at='b2tr'):
        with FLConfigHolder(llm) as ch:
            llm.fl_config.attack_mode = at
            llm.fl_config.collect_intermediates = False
            llm.fl_config.noise_mode = 'none'
            ch.change_config()

            if dummy_init is not None:
                dummy = dummy_init.clone().detach().to(llm.device).argmax(-1)
            else:
                dummy = torch.randint(0, llm.config.vocab_size, inter.fx.shape[:-1]).to(llm.device)
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
                dmy = dummy
                if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                    dmy = dummy.half()
                it = llm(inputs_embeds=dmy)
                target = inter.fx.to(llm.device)
                if it.dtype == torch.float16:
                    it = it.float()
                if target.dtype == torch.float16:
                    target = target.float()
                if dummy.dtype == torch.float16:
                    dummy = dummy.float()
                if attacker is not None:
                    out = attacker(it)
                    out2 = attacker(target)
                    loss = torch.nn.CrossEntropyLoss()(out, out2)
                else:
                    if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                        target = target.permute(1, 0, 2).contiguous()
                        it = it.permute(1, 0, 2).contiguous()
                    loss = 0
                    if cos_loss:
                        for x, y in zip(it, target):
                            loss += 1 - torch.cosine_similarity(x, y, dim=-1).mean()
                            # loss += ((x - y) ** 2).mean()# + 0.1 * torch.abs(x - y).mean().float()
                    else:
                        for x, y in zip(it, target):
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

    def fit_eia(self, tok, llm: SplitWrapperModel, inter: Intermediate, mapper: LMMapper, gt, mapped_to=-1, epochs=10,
                lr=1e-4,
                dummy_init=None,
                at='b2tr', temp=0.1, wd=0.01):
        with FLConfigHolder(llm) as ch:
            llm.fl_config.attack_mode = at
            llm.fl_config.collect_intermediates = False
            llm.fl_config.noise_mode = 'none'
            if mapper and mapped_to > 0:
                llm.fl_config.split_point_1 = mapped_to
            ch.change_config()
            if dummy_init is not None:
                dummy = dummy_init.clone().detach().to(llm.device)  # (batch_size, seq_len, vocab_size)
            else:
                dummy = torch.rand((inter.fx.shape[0], inter.fx.shape[1], llm.config.vocab_size)).to(llm.device)
                if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                    dummy = dummy.permute(1, 0)
            # dummy = torch.softmax(dummy / temp, -1)  # (batch_size, seq_len, vocab_size)
            pbar = tqdm(total=epochs)
            avg_rglf = 0
            avg_step = 0
            target = inter.fx.to(llm.device)
            if mapper:
                target = mapper(target).detach()
            dummy.requires_grad = True
            opt = torch.optim.AdamW([dummy], lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=wd)
            embedding_matrix = get_embedding_matrix(llm).float()  # (vocab, embed_size)
            for e in range(epochs):
                opt.zero_grad()
                dmy = torch.softmax(dummy / temp, -1) @ embedding_matrix
                it = llm(inputs_embeds=dmy)
                # if it.dtype == torch.float16:
                #     it = it.float()
                # if target.dtype == torch.float16:
                #     target = target.float()
                # if dummy.dtype == torch.float16:
                #     dummy = dummy.float()

                if isinstance(llm, ChatGLMForConditionalGenerationSplit):
                    target = target.permute(1, 0, 2).contiguous()
                    it = it.permute(1, 0, 2).contiguous()
                loss = 0
                for x, y in zip(it, target):
                    loss += ((x - y) ** 2).sum()  # + 0.1 * torch.abs(x - y).sum().float()
                loss.backward()
                opt.step()
                # print(f"Loss:{loss.item()} Before: {sent_before} After: {sent_after}")
                if e % 10 == 0:
                    rg, _, _ = evaluate_attacker_rouge(tok, dummy, gt)
                    avg_rglf += rg["rouge-l"]["f"]
                    avg_step += 1

                pbar.set_description(
                    f'Epoch {e}/{epochs} Loss: {loss.item()} ROUGE: {0 if avg_step == 0 else avg_rglf / avg_step}')
                pbar.update(1)
        return dummy

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
        return torch.softmax(cosine_similarities, -1)
