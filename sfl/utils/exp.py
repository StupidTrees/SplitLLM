import argparse
import dataclasses
import os
from copy import deepcopy

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, ViTImageProcessor

from sfl import config
from sfl.config import FLConfig, model_download_dir, reducer_path
from sfl.data.base import MixtureFedDataset, FedDataset
from sfl.model.llm.dim_reduction import DimReduction
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.vit.vit_wrapper import ViTForImageClassificationSplit
from sfl.simulator.param_keeper import InMemoryParameterKeeper
from sfl.utils.argparser import PrefixArgumentParser
from sfl.utils.model import get_best_gpu

_dataset_name_map = {}
_model_name_map = {}
_model_name_prefix_map = {}
_model_requiring_quantization = set()


def register_dataset(name):
    def wrapper(cls):
        assert issubclass(
            cls, FedDataset
        ), "All dataset must inherit FedDataset"
        _dataset_name_map[name] = cls
        return cls

    return wrapper


def register_model(names, register_for_prefix=True, requiring_quantization=False):
    dct = _model_name_map
    if register_for_prefix:
        dct = _model_name_prefix_map

    def wrapper(cls):
        assert issubclass(
            cls, SplitWrapperModel
        ), "All outer model must inherit SplitWrapperModel"
        if isinstance(names, str):
            dct[names] = cls
        elif isinstance(names, (list, tuple, set)):
            for n in names:
                dct[n] = cls
        else:
            raise ValueError("names must be str or list|tuple|set")

        if requiring_quantization:
            _model_requiring_quantization.add(cls)

        return cls

    return wrapper


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


def add_train_dra_params(parser):
    parser.add_argument('--exp_name', type=str, default='attacker')
    parser.add_argument('--model_download_dir', type=str, default='/root/autodl-tmp/sfl/models')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--save_dir', type=str, default=config.attacker_path)
    parser.add_argument('--dataset', type=str, default='piqa')
    parser.add_argument('--dataset_train_frac', type=float, default=1.0)
    parser.add_argument('--dataset_test_frac', type=float, default=1.0)
    parser.add_argument('--attack_model', type=str, default='moe', help='lstm or ...')
    parser.add_argument('--sps', type=str, default='6-26', help='split points')
    parser.add_argument('--attack_mode', type=str, default='tr2t', help='b2tr or t2tr')
    parser.add_argument('--load_bits', type=int, default=8, help='load bits for large models')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--epochs_gating', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--md_n_layers', type=int, default=2)
    parser.add_argument('--require_prefix', type=str, default=None, help='default is None, specify for existence check')
    parser.add_argument('--noise_mode', type=str, default='none')
    parser.add_argument('--noise_scale_dxp', type=float, default=0.0)
    parser.add_argument('--noise_scale_gaussian', type=float, default=0.0)
    parser.add_argument('--noise_scale_dc', type=float, default=0.0)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--save_threshold', type=float, default=0.1)
    parser.add_argument('--log_to_wandb', type=str2bool, default=False)
    parser.add_argument('--skip_exists', type=str2bool, default=True)


def add_train_mapper_params(parser):
    parser.add_argument('--exp_name', type=str, default='attacker')
    parser.add_argument('--model_download_dir', type=str, default='/root/autodl-tmp/sfl/models')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--save_dir', type=str, default=config.mapper_path)
    parser.add_argument('--dataset', type=str, default='piqa')
    parser.add_argument('--dataset_train_frac', type=float, default=1.0)
    parser.add_argument('--dataset_test_frac', type=float, default=1.0)
    parser.add_argument('--attack_model', type=str, default='moe', help='lstm or ...')
    parser.add_argument('--load_bits', type=int, default=8, help='load bits for large models')
    parser.add_argument('--target', type=str, default='6-1', help='mapping target layers')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)
    parser.add_argument('--checkpoint_freq', type=int, default=2)
    parser.add_argument('--save_threshold', type=float, default=0.1)
    parser.add_argument('--log_to_wandb', type=str2bool, default=False)
    parser.add_argument('--skip_exists', type=str2bool, default=True)


def add_train_reducer_params(parser):
    parser.add_argument('--exp_name', type=str, default='attacker')
    parser.add_argument('--model_download_dir', type=str, default='/root/autodl-tmp/sfl/models')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--save_dir', type=str, default=config.reducer_path)
    parser.add_argument('--dataset', type=str, default='piqa')
    parser.add_argument('--dataset_train_label', type=str, default='train')
    parser.add_argument('--dataset_train_frac', type=float, default=0.1)
    parser.add_argument('--dataset_test_frac', type=float, default=1.0)
    parser.add_argument('--attack_model', type=str, default='moe', help='lstm or ...')
    parser.add_argument('--load_bits', type=int, default=8, help='load bits for large models')
    parser.add_argument('--layer', type=int, default=6, help='target layer, 6 means output of #5 layer will be reduced')
    parser.add_argument('--alpha', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)
    parser.add_argument('--checkpoint_freq', type=int, default=2)
    parser.add_argument('--save_threshold', type=float, default=20.0)
    parser.add_argument('--log_to_wandb', type=str2bool, default=False)
    parser.add_argument('--skip_exists', type=str2bool, default=True)


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
    parser.add_argument('--noise_scale_dxp', type=float, default=0.0)
    parser.add_argument('--noise_scale_gaussian', type=float, default=0.0)
    parser.add_argument('--noise_scale_grad', type=float, default=0.0)
    parser.add_argument('--noise_scale_dc', type=float, default=0.1)
    parser.add_argument('--noise_scale_dc_sim', type=int, default=10)
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
    parser.add_argument('--reducer_train_frac', type=float, default=1.0)
    parser.add_argument('--reducer_train_label', type=str, default='train')
    parser.add_argument('--reducer_dataset', type=str,
                        default='')
    parser.add_argument('--reducer_layer', type=int,
                        default=-1)
    parser.add_argument('--reducer_alpha', type=int,
                        default=512)
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
                      noise_scale_dxp=args.noise_scale_dxp,  # 噪声大小
                      noise_scale_gaussian=args.noise_scale_gaussian,  # 噪声大小
                      noise_scale_grad=args.noise_scale_grad,
                      noise_scale_dc=args.noise_scale_dc,
                      noise_scale_dc_sim=args.noise_scale_dc_sim,
                      collect_intermediates=True,
                      collect_all_layers=args.collect_all_layers,
                      dataset_type=args.dataset_label,
                      batch_size=args.batch_size,
                      lr=args.lr
                      )
    return config


def get_model_path(model_name):
    path = ''
    if model_name.startswith('gpt2'):
        path = os.path.join(model_download_dir, model_name)
    elif model_name.startswith('bert'):
        path = os.path.join(model_download_dir, f"google-bert/{model_name}-uncased/")
    elif model_name.startswith('roberta'):
        path = os.path.join(model_download_dir, f"FacebookAI/{model_name}/")
    elif model_name.startswith('flan-t5'):
        path = os.path.join(model_download_dir, f"google/{model_name}")
    elif model_name.startswith('flan-ul2'):
        path = os.path.join(model_download_dir, f"google/{model_name}")
    elif model_name.startswith('llama2-raw'):
        path = os.path.join(model_download_dir, f"meta-llama/Llama-2-7b")
        # path = os.path.join(model_download_dir, f"daryl149/llama-2-7b-chat-hf")
    elif model_name.startswith('llama2'):
        path = os.path.join(model_download_dir, f"meta-llama/Llama-2-7b-chat-hf")
        # path = os.path.join(model_download_dir, f"daryl149/llama-2-7b-chat-hf")
    elif model_name.startswith('llama3'):
        path = os.path.join(model_download_dir, f"meta-llama/Meta-Llama-3-8B")
        # path = os.path.join(model_download_dir, f"daryl149/llama-2-7b-chat-hf")
    elif model_name.startswith('vicuna'):
        path = os.path.join(model_download_dir, f"lmsys/vicuna-7b-v1.5")
    elif model_name.startswith('chatglm'):
        path = os.path.join(model_download_dir, f"THUDM/chatglm3-6b")
    elif model_name.startswith('wizard'):
        path = os.path.join(model_download_dir, f"lucyknada/microsoft_WizardLM-2-7B")
    elif model_name.startswith('gptj'):
        path = os.path.join(model_download_dir, f"EleutherAI/gpt-j-6b")
    elif model_name.startswith('falcon'):
        path = os.path.join(model_download_dir, f"tiiuae/falcon-7b-instruct")
    elif model_name.startswith('codegen'):
        path = os.path.join(model_download_dir, f"Salesforce/codegen25-7b-instruct_P")
    elif model_name.startswith('bloomz'):
        path = os.path.join(model_download_dir, f"bigscience/bloomz-560m")
    return path


def get_tokenizer(model_name='gpt2'):
    return AutoTokenizer.from_pretrained(get_model_path(model_name), trust_remote_code=True)


@dataclasses.dataclass
class ReducerArgument:
    enable: bool = False
    dataset: str = None
    target_model: str = None
    train_label: str = 'train'
    train_frac: float = 1.0
    layer: int = 6
    alpha: int = 128


def get_reducer_args() -> ReducerArgument:
    parser = PrefixArgumentParser(prefix='reducer', dataclass_types=[ReducerArgument])
    return parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]


def get_dim_reducer(args, aargs: ReducerArgument):
    dataset = aargs.dataset
    if dataset is None:
        dataset = args.dataset
    model_name = aargs.target_model
    if model_name is None:
        model_name = args.model_name
    if required_quantization(args.model_name):
        model_name += f"-{args.load_bits}bits"
    mapper_path = reducer_path + f'{model_name}/{dataset}/'
    matches = []
    for d in os.listdir(mapper_path):
        pattern = f'{aargs.train_label}*{aargs.train_frac:.3f}'
        if ',' in dataset:
            pattern = f'Tr{aargs.train_frac:.3f}'
        if d.startswith(pattern):
            mapper_path = os.path.join(mapper_path, d) + '/'
            matches.append(mapper_path)
    assert len(matches) > 0
    mapper_path_1 = None
    for attacker_path in matches:
        mapper_path_1 = attacker_path + f'layer{aargs.layer}/{aargs.alpha}'
        l = sorted(list(os.listdir(mapper_path_1)), key=lambda x: float(x.split('_')[-1]))[0]
        mapper_path_1 = os.path.join(mapper_path_1, l)
        if not os.path.exists(mapper_path_1):
            mapper_path_1 = None
    if mapper_path_1:
        return DimReduction.from_pretrained(mapper_path_1)
    return None


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


def get_dataset(dataset_name, tokenizer, client_ids=None, shrink_frac=1.0, completion_only=False) -> FedDataset:
    if client_ids is None:
        client_ids = []
    if ',' in dataset_name:
        dataset_names = dataset_name.split(',')
        dataset_classes = [get_dataset_class(dn) for dn in dataset_names]
        return MixtureFedDataset(tokenizer, client_ids, shrink_frac, dataset_names, dataset_classes)
    else:
        return get_dataset_class(dataset_name)(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=shrink_frac,
                                               completion_only=completion_only)


def get_dataset_class(dataset_name):
    if dataset_name not in _dataset_name_map:
        raise AttributeError
    return _dataset_name_map[dataset_name]
