import numpy as np
import torch
from rouge import Rouge


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
                      max_length=100, num_beams=6, no_repeat_ngram_size=2, early_stopping=True,
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
        scores = wordscores.sum(1)/wordscores.shape[1]
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
