import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pynvml
import torch
from rouge import Rouge
from torch.nn import CrossEntropyLoss


@dataclass
class Intermediate:
    fx: Any
    grad: Any | None = None
    type: str = 'normal'


def calculate_rouge(tok, logits, labels, print_comparison=False, is_tokens=False):
    my_rouge = Rouge()
    if not is_tokens:
        output_texts = [tok.decode(logits.argmax(dim=-1)[i], skip_special_tokens=True) for i in
                        range(len(logits))]
    else:
        output_texts = [tok.decode(s, skip_special_tokens=True) for s in logits]
    hyps_and_refs = zip(output_texts, labels)
    hyps, refs = zip(*hyps_and_refs)
    if print_comparison:
        for h, r in zip(hyps, refs):
            print(f'{r}==>{h}')
    try:
        result = my_rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)  # 取一个 batch 的平均
    except:
        result = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                  'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
    return result


def calculate_rouge_text(texts, labels, print_comparison=False):
    my_rouge = Rouge()
    hyps_and_refs = zip(texts, labels)
    hyps, refs = zip(*hyps_and_refs)
    if print_comparison:
        for h, r in zip(hyps, refs):
            print(f'{r}==>{h}')
    try:
        result = my_rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)  # 取一个 batch 的平均
    except:
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
    # preds = torch.cat(predss)
    model.train(True)
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
    model.train(False)
    if hasattr(model.config, 'n_positions'):
        max_length = model.config.n_positions
    elif hasattr(model.config, 'max_length'):
        max_length = model.config.max_length
    ppl = 0
    len = 0
    for batch in loader:
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
    model.train(False)
    acc = 0
    itm = 0
    for batch in loader:
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
