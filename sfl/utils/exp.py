import argparse
import os
from copy import deepcopy

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, ViTImageProcessor

from sfl.config import FLConfig, model_download_dir
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.vit.vit_wrapper import ViTForImageClassificationSplit
from sfl.simulator.param_keeper import InMemoryParameterKeeper
from sfl.utils.model import get_best_gpu

_dataset_name_map = {}
_dataset_dra_train_label_map = {}
_dataset_dra_test_label_map = {}
_model_name_map = {}
_model_name_prefix_map = {}
_model_dir_name_map = {}
_model_dir_name_prefix_map = {}
_model_requiring_quantization = set()


def register_dataset(name, dra_train_label='validation', dra_test_label='test'):
    from sfl.data.base import FedDataset
    def wrapper(cls):
        assert issubclass(
            cls, FedDataset
        ), "All dataset must inherit FedDataset"
        _dataset_name_map[name] = cls
        _dataset_dra_train_label_map[name] = dra_train_label
        _dataset_dra_test_label_map[name] = dra_test_label
        return cls

    return wrapper


def register_model(names, register_for_prefix=True, requiring_quantization=False, dir_names=None):
    dct = _model_name_map
    dct_dir = _model_dir_name_map
    if register_for_prefix:
        dct = _model_name_prefix_map
        dct_dir = _model_dir_name_prefix_map
    dir = dir_names
    if dir is None:
        dir = names
    if isinstance(str, (list, tuple, set)):
        assert isinstance(dir, (list, tuple, set)) and len(dir) == len(names)

    def wrapper(cls):
        assert issubclass(
            cls, SplitWrapperModel
        ), "All outer model must inherit SplitWrapperModel"
        if isinstance(names, str):
            dct[names] = cls
            dct_dir[names] = dir
        elif isinstance(names, (list, tuple, set)):
            for n, dn in zip(names, dir):
                dct[n] = cls
                dct_dir[n] = dn
        else:
            raise ValueError("names must be str or list|tuple|set")

        if requiring_quantization:
            _model_requiring_quantization.add(cls)

        return cls

    return wrapper


def get_dra_train_label(name):
    return _dataset_dra_train_label_map[name]


def get_dra_test_label(name):
    return _dataset_dra_train_label_map[name]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_to_dict(args_list):
    dict = {}
    key = None
    for l in args_list:
        if key:
            dict[key] = l
            key = None
        elif l.startswith('--'):
            key = l[2:]
    return dict


def merge_args(args1, args2):
    for k, v in vars(args2).items():
        if not hasattr(args1, k):
            setattr(args1, k, v)
    return args1


def add_sfl_params(parser):
    parser.add_argument('--exp_name', type=str, default='compare_tag')
    parser.add_argument('--case_name', type=str, default='')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--load_bits', type=int, default=8, help='load bits for large models')
    parser.add_argument('--pre_ft_dataset', type=str, default='')
    parser.add_argument('--pre_ft_data_label', type=str, default='train')
    parser.add_argument('--pre_ft_max_steps', type=int, default=800)
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--dataset_max_seq_len', type=int, default=-1)
    parser.add_argument('--max_global_step', type=int, default=-1)
    parser.add_argument('--dataset_label', type=str, default='train')
    parser.add_argument('--data_shrink_frac', type=float, default=0.15, help='shrink data to this fraction')
    parser.add_argument('--test_data_label', type=str, default='test')
    parser.add_argument('--test_data_shrink_frac', type=float, default=0.15, help='shrink data to this fraction')
    parser.add_argument('--evaluate_freq', type=int, default=25)
    parser.add_argument('--collect_all_layers', type=str2bool, default=False,
                        help='collect intermediates of all layers')
    parser.add_argument('--split_points', type=str, default='6-30', help='split points, b2tr-tr2t')
    parser.add_argument('--lora_at_trunk', type=str2bool, default=True, help='use LoRA at trunk part')
    parser.add_argument('--lora_at_bottom', type=str2bool, default=False, help='use LoRA at bottom part')
    parser.add_argument('--lora_at_top', type=str2bool, default=False, help='use LoRA at top part')
    parser.add_argument('--lora_at_embed', type=str2bool, default=False, help='use LoRA at embedding layer')
    parser.add_argument('--noise_mode', type=str, default='none')
    parser.add_argument('--task_type', type=str, default=None)
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--client_num', type=int, default=1)
    parser.add_argument('--global_round', type=int, default=4)
    parser.add_argument('--client_from_scratch', type=str2bool, default=False)
    parser.add_argument('--self_pt_enable', type=str2bool, default=False)
    parser.add_argument('--entangle_enable', type=str2bool, default=False)
    parser.add_argument('--client_steps', type=int, default=50)
    parser.add_argument('--client_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_to_wandb', type=str2bool, default=True)
    parser.add_argument('--attacker_freq', type=int, default=25, help='attack every * steps')
    parser.add_argument('--attacker_samples', type=int, default=10, help='attack how many batches each time')
    parser.add_argument('--reducer_enable', type=str2bool, default=False)
    parser.add_argument('--completion_only', type=str2bool, default=False)


def get_fl_config(args) -> FLConfig:
    config = FLConfig(global_round=args.global_round,
                      client_evaluate_freq=args.evaluate_freq,
                      client_steps=args.client_steps,
                      client_epoch=args.client_epoch,  # 每轮联邦每个Client训x轮
                      max_global_step=args.max_global_step,
                      split_point_1=int(args.split_points.split('-')[0]),
                      split_point_2=int(args.split_points.split('-')[1]),
                      use_lora_at_trunk=args.lora_at_trunk,  # 在trunk部分使用LoRA
                      use_lora_at_top=args.lora_at_top,
                      use_lora_at_bottom=args.lora_at_bottom,
                      use_lora_at_embed=args.lora_at_embed,
                      top_and_bottom_from_scratch=args.client_from_scratch,  # top和bottom都不采用预训练参数.
                      noise_mode=args.noise_mode,
                      noise_scale=args.noise_scale,
                      collect_intermediates=True,
                      collect_all_layers=args.collect_all_layers,
                      dataset_type=args.dataset_label,
                      batch_size=args.batch_size,
                      reducer_enable=args.reducer_enable,
                      lr=args.lr
                      )
    return config


def get_model_path(model_name):
    dn = None
    if model_name in _model_dir_name_map:
        dn = _model_dir_name_map[model_name]
    else:
        for prefix in _model_dir_name_prefix_map:
            if model_name.startswith(prefix):
                dn = _model_dir_name_prefix_map[prefix]
                break
        if dn is None:
            raise AttributeError(f"Path for Model {model_name} not found")
    dn = dn.replace('$model_name', model_name)
    path = os.path.join(model_download_dir, dn)
    return path


def get_tokenizer(model_name='gpt2'):
    return AutoTokenizer.from_pretrained(get_model_path(model_name), trust_remote_code=True)


def required_quantization(model_name):
    model_cls = get_model_class(model_name)
    return model_cls in _model_requiring_quantization


def load_model_in_param_keepers(model_name, fl_config, parts=None):
    """
    Load the pre-trained weights of the model into the parameter keeper.
    :return: parameter keeper
    """
    cross_model, _ = get_model_and_tokenizer(model_name)
    cross_model.config_sfl(fl_config)
    cross_model = cross_model.convert_to_lora_model()
    pk = InMemoryParameterKeeper([])
    if parts is None:
        parts = ['bottom']
    for key in ['pretrained']:
        for part in parts:
            if part == 'bottom':
                params = cross_model.get_bottom_params()
            elif part == 'top':
                params = cross_model.get_top_params()
            elif part == 'trunk':
                params = cross_model.get_trunk_params()
            pk.store_other_params(key, part,
                                  [deepcopy(p.detach().cpu()) for nm, p in params])
    del cross_model
    return pk


def get_model_class(model_name):
    if model_name in _model_name_map:
        clz = _model_name_map[model_name]
    else:
        clz = None
        for prefix in _model_name_prefix_map:
            if model_name.startswith(prefix):
                clz = _model_name_prefix_map[prefix]
                break
        if clz is None:
            raise AttributeError(f"Model {model_name} not found")
    return clz


def get_model(model_name='gpt2', task='lm', num_labels=2, tokenizer=None, load_bits=8, force_on_best_gpu=True,
              do_not_specify_device_map=False,
              **kwargs):
    clz = get_model_class(model_name)
    if required_quantization(model_name):
        if load_bits <= 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=load_bits == 8,  # load the model into memory using 8-bit precision
                load_in_4bit=load_bits == 4,  # load the model into memory using 4-bit precision
                bnb_4bit_use_double_quant=True,  # use double quantition
                bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_8bit_use_double_quant=True,  # use double quantition
                bnb_8bit_quant_type="nf8",  # use NormalFloat quantition
                bnb_8bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
            )
            kwargs['quantization_config'] = bnb_config

    if task == 'clsf':
        kwargs['num_labels'] = num_labels
    if force_on_best_gpu:
        device_map = str(get_best_gpu())
    else:
        device_map = 'auto'
    if do_not_specify_device_map:
        model = clz.from_pretrained(get_model_path(model_name), **kwargs)
    else:
        model = clz.from_pretrained(get_model_path(model_name), device_map=device_map, **kwargs)
    if tokenizer is not None and 'chatglm' not in model_name:
        if model.config.pad_token_id is not None:
            tokenizer.pad_token_id = model.config.pad_token_id
        if model.config.eos_token_id is not None:
            tokenizer.pad_token_id = model.config.eos_token_id
    return model


def get_model_and_tokenizer(model_name='gpt2', task='lm', num_labels=2, force_on_best_gpu=True,
                            do_not_specify_device_map=False,
                            **kwargs):
    if model_name.startswith('vit'):
        processor = ViTImageProcessor.from_pretrained(
            os.path.join(model_download_dir, f'google/{model_name}-patch16-224'))
        model = ViTForImageClassificationSplit.from_pretrained(
            os.path.join(model_download_dir, f'google/{model_name}-patch16-224'))
        return model, processor
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, task, num_labels, tokenizer, force_on_best_gpu=force_on_best_gpu,
                      do_not_specify_device_map=do_not_specify_device_map, **kwargs)

    return model, tokenizer


def get_dataset(dataset_name, tokenizer, client_ids=None, shrink_frac=1.0, completion_only=False):
    if client_ids is None:
        client_ids = []
    if ',' in dataset_name:
        dataset_names = dataset_name.split(',')
        dataset_classes = [get_dataset_class(dn) for dn in dataset_names]
        from sfl.data.base import MixtureFedDataset
        return MixtureFedDataset(tokenizer, client_ids, shrink_frac, dataset_names, dataset_classes)
    else:
        return get_dataset_class(dataset_name)(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=shrink_frac,
                                               completion_only=completion_only)


def get_dataset_class(dataset_name):
    import sfl.data.datasets as datasets
    clz = datasets.FedDataset
    if dataset_name not in _dataset_name_map:
        raise AttributeError
    clz = _dataset_name_map[dataset_name]
    return clz
