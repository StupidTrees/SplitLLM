import argparse
import os
import random

import numpy as np
import pynvml
import torch
from torch.nn import CrossEntropyLoss

from sfl.model.attack_model import LSTMAttackModel, GRUAttackModel, TransformerEncoderAttackModel, LinearAttackModel
from sfl.simulator.dataset import PIQAFedDataset, GSM8KFedDataset


def get_dataset_class(dataset_name):
    dataset_cls = PIQAFedDataset
    if dataset_name == 'piqa':
        dataset_cls = PIQAFedDataset
    elif dataset_name:
        dataset_cls = GSM8KFedDataset
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


def extract_attacker_path(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    attacker_path = args.attacker_path + f'{args.model_name}/{args.attacker_dataset}/'
    # find the first directory in  attacker_path starts with {args.attacker_train_label}*{args.attacker_trian_frac}
    for d in os.listdir(attacker_path):
        if d.startswith(f'{args.attacker_train_label}*{args.attacker_train_frac:.3f}'):
            attacker_path = os.path.join(attacker_path, d)
            break
    attacker_path += f'/{args.attack_model}'
    attacker_path_1 = attacker_path + f'/b2tr-{args.split_point_1}/' + args.attacker_prefix
    attacker_path_2 = attacker_path + f'/tr2t-{args.split_point_2}/' + args.attacker_prefix
    # list all dirs under attacker_path_1
    l = sorted(list(os.listdir(attacker_path_1)), key=lambda x: float(x.split('_')[-1]))[-1]
    attacker_path_1 = os.path.join(attacker_path_1, l)
    l = sorted(list(os.listdir(attacker_path_2)), key=lambda x: float(x.split('_')[-1]))[-1]
    attacker_path_2 = os.path.join(attacker_path_2, l)
    return attacker_path_1, attacker_path_2


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
