import argparse
import os
import random

import numpy as np
import pynvml
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

import sfl
from sfl.model.attack_model import LSTMAttackModel, GRUAttackModel, TransformerEncoderAttackModel, LinearAttackModel
from sfl.model.bert.bert_wrapper import BertForSequenceClassificationSplitModel
from sfl.model.gpt2.gpt2_clsf import GPT2SplitClassificationModel
from sfl.simulator.dataset import PIQAFedDataset, GSM8KFedDataset, WikiTextFedDataset, CodeAlpacaFedDataset, \
    DialogSumFedDataset, IMDBFedDataset
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel


def get_model_path(model_name):
    path = ''
    if model_name == 'gpt-2':
        path = os.path.join(sfl.config.model_download_dir, model_name)
    elif model_name == 'bert':
        path = os.path.join(sfl.config.model_download_dir, "google-bert/bert-base-uncased/")
    return path


def get_tokenizer(model_name='gpt-2'):
    return AutoTokenizer.from_pretrained(get_model_path(model_name))


def get_model(model_name='gpt-2', task='lm', num_labels=2, tokenizer=None):
    clz = GPT2SplitLMHeadModel
    kwargs = {}
    if model_name == 'gpt-2':
        if task == 'lm':
            clz = GPT2SplitLMHeadModel
        elif task == 'clsf':
            clz = GPT2SplitClassificationModel
    elif model_name == 'bert':
        clz = BertForSequenceClassificationSplitModel
    if task == 'clsf':
        kwargs['num_labels'] = num_labels
    model = clz.from_pretrained(get_model_path(model_name), **kwargs)
    if tokenizer is not None:
        if model.config.pad_token_id is not None:
            tokenizer.pad_token_id = model.config.pad_token_id
        if model.config.eos_token_id is not None:
            tokenizer.pad_token_id = model.config.eos_token_id
    return model


def get_model_and_tokenizer(model_name='gpt-2', task='lm', num_labels=2):
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, task, num_labels, tokenizer)
    return model, tokenizer


def get_dataset_class(dataset_name):
    if dataset_name == 'piqa':
        dataset_cls = PIQAFedDataset
    elif dataset_name == 'gsm8k':
        dataset_cls = GSM8KFedDataset
    elif dataset_name == 'wikitext':
        dataset_cls = WikiTextFedDataset
    elif dataset_name == 'codealpaca':
        dataset_cls = CodeAlpacaFedDataset
    elif dataset_name == 'dialogsum':
        dataset_cls = DialogSumFedDataset
    elif dataset_name == 'imdb':
        dataset_cls = IMDBFedDataset
    else:
        raise AttributeError
    return dataset_cls


def get_attacker_class(attack_model):
    attacker_cls = LSTMAttackModel
    if attack_model == 'lstm':
        attacker_cls = LSTMAttackModel
    elif attack_model == 'gru':
        attacker_cls = GRUAttackModel
    elif attack_model == 'linear':
        attacker_cls = LinearAttackModel
    elif attack_model == 'trans-enc':
        attacker_cls = TransformerEncoderAttackModel
    return attacker_cls


def extract_attacker_path(args, attacker_cls):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    attacker_path = args.attacker_path + f'{args.model_name}/{args.attacker_dataset}/'
    # find the first directory in  attacker_path starts with {args.attacker_train_label}*{args.attacker_trian_frac}
    match = False
    for d in os.listdir(attacker_path):
        if d.startswith(f'{args.attacker_train_label}*{args.attacker_train_frac:.3f}'):
            attacker_path = os.path.join(attacker_path, d) + '/'
            match = True
            break
    assert match
    attacker_path += f'{args.attacker_model}'
    attacker_path_1 = None
    if args.attacker_b2tr_enable:
        sp1 = int(args.split_points.split('-')[0])
        if args.attacker_b2tr_sp >= 0:
            sp1 = args.attacker_b2tr_sp
        attacker_path_1 = attacker_path + f'/b2tr-{sp1}/' + args.attacker_prefix
        l = sorted(list(os.listdir(attacker_path_1)), key=lambda x: float(x.split('_')[-1]))[-1]
        attacker_path_1 = os.path.join(attacker_path_1, l)
    attacker_path_2 = None
    if args.attacker_tr2t_enable:
        sp2 = int(args.split_points.split('-')[1])
        if args.attacker_tr2t_sp >= 0:
            sp2 = args.attacker_tr2t_sp
        attacker_path_2 = attacker_path + f'/tr2t-{sp2}/' + args.attacker_prefix
        if not os.path.exists(attacker_path_2):
            attacker_path_2 = attacker_path_2.replace('tr2t', 'b2tr')
        l = sorted(list(os.listdir(attacker_path_2)), key=lambda x: float(x.split('_')[-1]))[-1]
        attacker_path_2 = os.path.join(attacker_path_2, l)
    attacker, attacker2 = None, None
    if attacker_path_1:
        attacker = attacker_cls.from_pretrained(attacker_path_1)
    if attacker_path_2:
        attacker2 = attacker_cls.from_pretrained(attacker_path_2)
    return attacker, attacker2


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
    max_length = model.config.n_positions
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
