import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pynvml
import torch
from PIL.Image import fromarray
from matplotlib import pyplot
from rouge import Rouge
from torch.nn import CrossEntropyLoss, Parameter
from trl import DPOTrainer

from sfl.config import dxp_moe_range, gaussian_moe_range, dc_moe_range
from sfl.model.llm.glm import modeling_chatglm


@dataclass
class Intermediate:
    fx: Any
    grad: Any | None = None
    type: str = 'normal'


def decode_with_extra_space(tok, sent):
    strs = []
    for tid in sent:
        strs.append(tok.decode(tid, skip_special_tokens=True))
    # print(' '.join(strs)[:20], tok.decode(sent, skip_special_tokens=True)[:20])
    return ' '.join(strs)


def evaluate_attacker_rouge(tok, attacker_logits, batch):
    if 'input_santi_mask' in batch:  # 只评估敏感词
        mask = batch['input_santi_mask']
        zero_indexes = torch.where(mask == 0)
        masked_atk_ids = attacker_logits.argmax(dim=-1)
        masked_atk_ids[zero_indexes] = tok.eos_token_id
        masked_gt_ids = batch['input_ids'].clone()
        masked_gt_ids[zero_indexes] = tok.eos_token_id
        atk_txts = [decode_with_extra_space(tok, s) for s in masked_atk_ids]
        gt_txts = [decode_with_extra_space(tok, s) for s in masked_gt_ids]
    else:
        atk_txts = [decode_with_extra_space(tok, s) for s in attacker_logits.argmax(dim=-1)]
        gt_txts = [decode_with_extra_space(tok, s) for s in batch['input_ids']]  # batch['input_text']
    rouge = calculate_rouge_text(atk_txts, gt_txts, print_comparison=False)
    meteor = calculate_meteor(atk_txts, gt_txts)
    token_acc = calculate_token_acc(tok, atk_txts, gt_txts)
    return rouge, meteor, token_acc


def calculate_token_acc(tok, texts, labels):
    f1_avg = 0
    for g, r in zip(texts, labels):
        generated_tokens = set(tok.tokenize(g))
        reference_tokens = set(tok.tokenize(r))
        # calculate overlap number
        overlap = len(generated_tokens & reference_tokens)
        # calculate precision and recall
        precision = overlap / len(generated_tokens)
        recall = overlap / len(reference_tokens)
        # calculate f1
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_avg += f1
    if len(texts) == 0:
        return 0
    return f1_avg / len(texts)


def evaluate_attacker_mse(attacker_outputs, batch):
    return torch.nn.functional.mse_loss(attacker_outputs, batch)


def calculate_rouge(tok, logits, labels, print_comparison=False, is_tokens=False):
    if not is_tokens:
        output_texts = [tok.decode(logits.argmax(dim=-1)[i], skip_special_tokens=True) for i in
                        range(len(logits))]
    else:
        output_texts = [tok.decode(s, skip_special_tokens=True) for s in logits]
    return calculate_rouge_text(output_texts, labels, print_comparison=print_comparison)


import nltk

nltk.download('wordnet')
from nltk.translate import meteor_score


def calculate_meteor(texts, labels):
    meteor_sum = 0
    meteor_num = 0
    for text, label in zip(texts, labels):
        meteor = meteor_score.meteor_score([label.split()], text.split())
        meteor_sum += meteor
        meteor_num += 1
    return meteor_sum / meteor_num


def calculate_rouge_text(texts, labels, print_comparison=False):
    my_rouge = Rouge()
    hyps_and_refs = zip(texts, labels)
    hyps, refs = zip(*hyps_and_refs)
    if print_comparison:
        for h, r in zip(hyps, refs):
            print(f'{r}==>{h}')
    try:
        hyps = list('<EMP>' if len(h) == 0 else h for h in hyps)
        result = my_rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)  # 取一个 batch 的平均
    except Exception as e:
        print(f'rouge error {e}')
        print(hyps, refs)
        result = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
    return result


# 测试模型的生成文本
def generate(text, tokenizer, md):
    md.train(False)
    t = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    res = md.generate(t['input_ids'].to(md.device), attention_mask=t['attention_mask'].to(md.device),
                      max_length=512, num_beams=6, no_repeat_ngram_size=2, early_stopping=True,
                      num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(res[0], skip_special_tokens=True)


# 测试模型输出
def get_output(text, tokenizer, md):
    t = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    if md.type == 'encoder-decoder':
        res = md(t['input_ids'].to(md.device), attention_mask=t['attention_mask'].to(md.device),
                 decoder_input_ids=t['input_ids'].to(md.device))
    else:
        res = md(t['input_ids'].to(md.device), attention_mask=t['attention_mask'].to(md.device))

    r = tokenizer.decode(res.logits.argmax(dim=-1)[-1], skip_special_tokens=True)
    return r


def sentence_score(sentence, model, tokenizer):
    sentence = torch.LongTensor(tokenizer.encode(sentence))
    before = torch.LongTensor(
        tokenizer.encode("<|endoftext|>")
    )  # ],return_tensors='pt').input_ids[0]
    after = torch.LongTensor(
        tokenizer.encode("<|endoftext|>")
    )  # ],return_tensors='pt').input_ids# [0]

    sent = sentence

    padded = torch.cat([before, sent, after], 0).unsqueeze(0).to(model.device)
    stride = 16
    padded = padded.long()
    wordscoress, scoress, predss = [], [], []
    for i in range(int(np.ceil(len(padded) / stride))):
        outputs = model(padded[i * stride: min((i + 1) * stride, len(padded))])
        lsm = -outputs[0].log_softmax(2)
        preds = torch.zeros_like(lsm)
        preds[:, 1:] = lsm[:, :-1]
        wordscores = (
            preds.gather(
                2, padded[i * stride: min((i + 1) * stride, len(padded))].unsqueeze(2)
            )
            .squeeze(2)
            .cpu()
            .detach()
        )
        scores = wordscores.sum(1) / wordscores.shape[1]
        wordscoress.append(wordscores)
        scoress.append(scores)
        predss.append(preds.cpu().detach())

    # wordscores = torch.cat(wordscoress)
    score = torch.cat(scoress).min()
    # preds = torch.cat(predss)
    return score


def sentence_score_tokens(sent, model):
    model.train(False)
    padded = sent.to(model.device).long()
    stride = 16
    scoress = []
    for i in range(int(np.ceil(len(padded) / stride))):
        outputs = model(padded[i * stride: min((i + 1) * stride, len(padded))])
        lsm = -outputs[0].log_softmax(2)
        preds = torch.zeros_like(lsm)
        preds[:, 1:] = lsm[:, :-1]
        wordscores = (
            preds.gather(
                2, padded[i * stride: min((i + 1) * stride, len(padded))].unsqueeze(2)
            )
            .squeeze(2)
            .detach()
        )
        scores = wordscores.sum(1) / wordscores.shape[1]
        scoress.append(scores)
    # wordscores = torch.cat(wordscoress)
    score = torch.cat(scoress)
    model.train(True)
    DPOTrainer
    return score


def get_best_gpu():
    """Return gpu (:class:`torch.device`) with largest free memory."""
    assert torch.cuda.is_available()
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys() is not None:
        cuda_devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        ]
    else:
        cuda_devices = range(deviceCount)

    assert max(cuda_devices) < deviceCount
    deviceMemory = []
    for i in cuda_devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))


def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def calc_unshift_loss(lm_logits, labels):
    labels = labels.to(lm_logits.device)
    # do not shift
    loss_fct = CrossEntropyLoss()
    return loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


def calc_unshift_loss_logits(lm_logits, labels):
    labels = labels.to(lm_logits.device)
    # do not shift
    loss_fct = CrossEntropyLoss()
    return loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1, labels.size(-1)))


def calc_shifted_loss_logits(lm_logits, label_logits):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = label_logits[..., 1:, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # return loss_fct(shift_logits, shift_labels)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1, shift_labels.size(-1)))


def calc_chatglm_loss(lm_logits, label_logits):
    lm_logits = lm_logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = label_logits[..., 1:, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1, shift_labels.size(-1)))


def calc_shifted_loss(lm_logits, labels):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def calc_circular_loss(lm_logits, labels):
    shift_logits = lm_logits[..., :, :].contiguous()
    shift_labels = torch.concat([labels[..., 1:], labels[..., 0:1]], dim=-1).contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


from tqdm import tqdm


def batched_perplexity(model, dataloader, tokenizer, stride=512):
    device = model.device
    max_len = model.config.n_positions
    # encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    # text_len = encodings.input_ids.size(1)
    lls = []
    ppl_total = 0
    batch_num = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        batch_size, text_len = input_ids.shape[:2]
        for i in tqdm(range(0, text_len, batch_size * stride)):
            begin_locs, end_locs, trg_lens = [], [], []
            for j in range(batch_size):
                j = i + j * stride
                if j >= text_len:
                    break
                begin_loc = max(j + stride - max_len, 0)
                end_loc = min(j + stride, text_len)
                trg_len = end_loc - j  # may be different from stride on last loop

                begin_locs.append(begin_loc)
                end_locs.append(end_loc)
                trg_lens.append(trg_len)

            input_ids = [input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]
            target_end_locs = [sen.size(-1) for sen in input_ids]
            input_ids = [
                torch.nn.functional.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
            ]  # we dont need attention mask as long as these padded token is not involved in loss calculation
            input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

            target_ids = torch.ones_like(
                input_ids) * -100  # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
            for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
                labels = input_ids[i, -b:e].clone()
                target_ids[i, -b:e] = labels

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs["loss"] * sum(trg_lens)

            lls.append(log_likelihood)
        ppl_total += torch.exp(sum(torch.stack(lls) / end_locs[-1]))
        batch_num += 1
    return ppl_total / batch_num


def evaluate_perplexity(model, loader, stride=1024):
    if loader is None:
        return 0
    model.train(False)
    if hasattr(model.config, 'n_positions'):
        max_length = model.config.n_positions
    elif hasattr(model.config, 'max_length'):
        max_length = model.config.max_length
    ppl = 0
    len = 0
    for batch in tqdm(loader, desc='evaluating perplexity'):
        input_ids = batch['input_ids'].to(model.device)
        seq_len = input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl += torch.exp(torch.stack(nlls).mean())
        len += 1
    model.train(True)
    return ppl / len


def evaluate_accuracy(model, loader):
    if loader is None:
        return 0
    model.train(False)
    acc = 0
    itm = 0
    for batch in loader:
        if 'input_ids' not in batch:
            input_ids = batch['input'].to(model.device)
        else:
            input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)  # (batch_size, )

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.view(-1, model.num_labels)  # (, num_labels)
            preds = torch.argmax(logits, dim=-1)
            labels = labels.view(-1)
            acc += (preds == labels).sum() / labels.numel()
            itm += 1
    model.train(True)
    return acc / itm


def get_t5_input(batch, tokenizer, device):
    q_ids = batch['q_ids'].to(device)
    q_mask = batch['q_att_mask'].to(device)
    a_ids = batch['a_ids'].to(device)
    dec_input = a_ids[:, :-1].contiguous()
    labels = a_ids[:, 1:].contiguous()
    labels[a_ids[:, 1:] == tokenizer.pad_token_id] = -100
    return {'input_ids': q_ids, 'attention_mask': q_mask, 'decoder_input_ids': dec_input, 'labels': labels}


def dist_corr(x, y):
    """
    Compute the distance correlation function
    """
    n = x.size(0)
    a = torch.cdist(x, x)
    b = torch.cdist(y, y)
    A = a - a.mean(dim=0) - a.mean(dim=1)[:, None] + a.mean()
    B = b - b.mean(dim=0) - b.mean(dim=1)[:, None] + b.mean()
    dcov2 = (A * B).sum() / n ** 2
    dvar = (A ** 2).sum() / n ** 2
    dvar2 = (B ** 2).sum() / n ** 2
    dcor = dcov2 / (torch.sqrt(dvar) * torch.sqrt(dvar2))
    return dcor


def random_choose_noise(input_scales=None, mode='dxp'):
    if input_scales is None:
        if mode == 'dxp':
            input_scales = dxp_moe_range
        elif mode == 'gaussian':
            input_scales = gaussian_moe_range
        elif mode == 'dc':
            input_scales = dc_moe_range
    scales = set()
    for s in input_scales:
        if s > 0:
            scales.add(s)
    numbers = [random.uniform(min(scales), max(scales)) for _ in
               range(len(scales))]
    numbers += [0, 0, 0, 0]
    plus_one = max(scales) * 2
    plus_zero = min(scales) / 2
    if mode == 'dxp' or mode == 'gaussian':
        numbers += [plus_one]
    if mode == 'dc':
        numbers += [plus_zero]
    return random.choice(numbers)


def convert_to_image(attacker_logits):
    recovered = (attacker_logits + 1) / 2
    return [fromarray((rec.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)) for rec in recovered]


def get_embedding_layer(llm):
    if hasattr(llm, 'model'):
        return llm.model.embed_tokens
    elif hasattr(llm, 'transformer'):
        if hasattr(llm.transformer, 'wte'):
            return llm.transformer.wte
        elif hasattr(llm.transformer, 'embedding'):
            return llm.transformer.embedding
    elif hasattr(llm, 'bert'):
        return llm.bert.embeddings.word_embeddings
    return None


def get_embedding_matrix(llm) -> Parameter:
    embedding = get_embedding_layer(llm)
    if isinstance(embedding, torch.nn.Embedding):
        return embedding.weight
    elif isinstance(embedding, modeling_chatglm.Embedding):
        return embedding.word_embeddings.weight



class FLConfigHolder:

    def __init__(self, llm):
        self.llm = llm
        self.config_bk = deepcopy(llm.fl_config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.llm.config_sfl(self.config_bk, None)

    def change_config(self):
        # change self.llm.fl_config's kw to value
        self.llm.config_sfl(self.llm.fl_config, None)
        # get an instance of fl_cnfig with its collect_intermediates set to False


def saliency_analysis_generative(llm, input_ids, max_length=32):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        llm.fl_config.noise_mode = 'none'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        generated = []
        for batch_iid in input_ids:
            iids = batch_iid.unsqueeze(0).to(llm.device)  # (1, seq_len)
            all_saliency = []
            cnt = 0
            while True:
                input_embeds = get_embedding_layer(llm)(iids)
                input_embeds.requires_grad = True
                outputs = llm(inputs_embeds=input_embeds)
                logits = outputs.logits  # (1,seq_len,vocab_size)
                next_token_id = logits.argmax(dim=-1)[:, -1]  # (1,)
                # logits = torch.max(logits[:, -1, :], dim=-1)
                # loss = logits.values.mean()
                loss = torch.max(logits[:, -1, :], dim=-1).values.mean()
                # loss = outputs.loss
                loss.backward()
                saliency = torch.abs(input_embeds.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                iids = torch.cat([iids, next_token_id.unsqueeze(0)], dim=-1)
                cnt += 1
                if next_token_id.item() == llm.config.eos_token_id or cnt > max_length:
                    break
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
            generated.append(iids[0, input_ids.size(1):])
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        # grads = input_embeds.grad  # (batch_size, seq_len, embed_size)

        # 获得重要性权重
        # pooled_grads = grads.mean(dim=1)  # (batch_size, embed_size)
        # features = input_embeds  # (batch_size, seq_len, embed_size)
        # # multiply the pooled gradients with the features on the last axis
        # saliency = torch.einsum("bse,be->bs", features, pooled_grads)  # (batch_size, seq_len)
        # saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
        # return weights
        llm.train(True)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, generated, saliency_stacks


def saliency_analysis_direct(llm, input_ids):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        llm.fl_config.noise_mode = 'none'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            for token_idx in range(batch_iid.size(-1)):
                input_embeds = get_embedding_layer(llm)(batch_iid.unsqueeze(0).to(llm.device))
                input_embeds.requires_grad = True
                outputs = llm(inputs_embeds=input_embeds)
                logits = outputs.logits  # (1,seq_len,vocab_size)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                # loss = outputs.loss
                loss.backward()
                saliency = torch.abs(input_embeds.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        llm.train(True)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, saliency_stacks


def saliency_analysis_decoder(llm, input_ids):
    llm.train(True)
    with FLConfigHolder(llm) as ch:
        bk_inter = {k: v for k, v in llm.intermediate_fx.items()}
        llm.fl_config.collect_intermediates = True
        llm.fl_config.noise_mode = 'none'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            for token_idx in range(batch_iid.size(-1)):
                outputs = llm(batch_iid.unsqueeze(0).to(llm.device))
                logits = outputs.logits  # (1,seq_len,vocab_size)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                # loss = outputs.loss
                loss.backward()
                b2tr_inter, _, _ = llm.get_all_inter()
                saliency = torch.abs(b2tr_inter.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
        llm.intermediate_fx = bk_inter
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, saliency_stacks


def saliency_analysis_atk(llm, attacker, input_ids):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        attacker.train(True)
        llm.fl_config.noise_mode = 'none'
        llm.fl_config.attack_mode = 'b2tr'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        attacked = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            for token_idx in range(batch_iid.size(-1)):
                input_embeds = get_embedding_layer(llm)(batch_iid.unsqueeze(0).to(llm.device))
                input_embeds.requires_grad = True
                inter = llm(inputs_embeds=input_embeds)
                logits = attacker(inter)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                loss.backward()
                saliency = torch.abs(input_embeds.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
            attacked.append(logits[0, :, :].argmax(dim=-1))
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        llm.train(True)
        attacker.train(False)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, attacked, saliency_stacks


def saliency_analysis_atk_mid(llm, attacker, input_ids):
    llm.train(False)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.collect_intermediates = False
        attacker.train(True)
        llm.fl_config.noise_mode = 'none'
        llm.fl_config.attack_mode = 'b2tr'
        ch.change_config()
        saliency_stacks = []
        saliency_avgs = []
        attacked = []
        for batch_iid in input_ids:
            all_saliency = []
            cnt = 0
            inter = llm(batch_iid.unsqueeze(0).to(llm.device))
            for token_idx in range(batch_iid.size(-1)):
                inter = inter.clone().detach().requires_grad_(True)
                logits = attacker(inter)
                loss = torch.max(logits[:, token_idx, :], dim=-1).values.mean()
                loss.backward()
                saliency = torch.abs(inter.grad).sum(dim=-1)  # (1, seq_len)
                saliency = saliency / saliency.max(dim=-1).values.unsqueeze(-1)
                all_saliency.append(saliency)
                cnt += 1
            all_saliency = [s[0, :batch_iid.size(-1)] for s in all_saliency]  # (1, seq_len)
            saliency_stack = torch.stack(all_saliency)  # (all, seq_len)
            saliency_stacks.append(saliency_stack.detach().cpu().numpy())
            saliency_avgs.append(saliency_stack.mean(dim=0))
            attacked.append(logits[0, :, :].argmax(dim=-1))
        batch_saliency_avg = torch.stack(saliency_avgs)  # (batch_size, seq_len)
        llm.train(True)
        attacker.train(False)
    saliency = batch_saliency_avg.detach().cpu().numpy()
    return saliency, attacked, saliency_stacks


def draw_saliency_map(saliency_matrix, input_sentence, output_sentence):
    # plot heatmap on saliency_matrix and log to wandb
    fig, ax = pyplot.subplots()
    fig.set_size_inches(16, 16)
    cax = ax.matshow(saliency_matrix, cmap='hot', vmin=0, vmax=1)
    # scale it to square
    ax.set_aspect('auto')
    fig.colorbar(cax)

    ax.set_xticks(ticks=range(len(input_sentence)), labels=input_sentence)
    ax.set_yticks(ticks=range(len(output_sentence)), labels=output_sentence)

    # ax.set_yticklabels(self.tokenizer.convert_ids_to_tokens(a), rotation=45)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('Saliency Matrix')
    return fig


def draw_generative_saliency_maps(tokenizer, input_sentence, next_token_ids, stacks):
    figs = []
    for i, (q, a, saliency_matrix) in enumerate(zip(input_sentence, next_token_ids, stacks)):
        fig = draw_saliency_map(saliency_matrix, tokenizer.convert_ids_to_tokens(q),
                                tokenizer.convert_ids_to_tokens(a))
        figs.append(fig)
        # close figure to avoid memory leak
    return figs


def draw_direct_saliency_maps(tokenizer, input_sentence, stacks):
    figs = []
    for i, (q, saliency_matrix) in enumerate(zip(input_sentence, stacks)):
        fig = draw_saliency_map(saliency_matrix, tokenizer.convert_ids_to_tokens(q),
                                tokenizer.convert_ids_to_tokens(q))
        figs.append(fig)
    return figs
