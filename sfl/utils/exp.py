import argparse
import dataclasses
import os

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, ViTImageProcessor

from sfl import config
from sfl.config import FLConfig, model_download_dir, mapper_path, reducer_path
from sfl.model.llm.bert.bert_wrapper import BertForSequenceClassificationSplitModel
from sfl.model.llm.dim_reduction import DimReduction
from sfl.model.llm.glm.glm_wrapper import ChatGLMForConditionalGenerationSplit
from sfl.model.llm.gpt2.gpt2_wrapper import GPT2SplitLMHeadModel, GPT2SplitClassificationModel
from sfl.model.llm.llama2.llama2_wrapper import LLAMA2SplitLMHeadModel
from sfl.model.llm.roberta.roberta_wrapper import RobertaForSequenceClassificationSplitModel
from sfl.model.llm.t5.t5wrapper import T5ForConditionalGenerationSplitModel
from sfl.model.llm.vit.vit_wrapper import ViTForImageClassificationSplit
from sfl.model.llm.wizard.wiard_wrapper import WizardForCausalLMSplit
from sfl.simulator.dataset import CodeAlpacaFedDataset, DialogSumFedDataset, IMDBFedDataset, PIQAFedDataset, \
    GSM8KFedDataset, WikiTextFedDataset, FedDataset, MixtureFedDataset, SensiMarkedFedDataset, \
    SensiReplacedFedDataset, SensiMaskedFedDataset, HC3CNFedDataset, ImageWoofFedDataset, PIQAMiniFedDataset
from sfl.utils.argparser import PrefixArgumentParser
from sfl.utils.model import get_best_gpu


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
    parser.add_argument('--data_shrink_frac', type=float, default=0.15, help='shrink dataset to this fraction')
    parser.add_argument('--test_data_label', type=str, default='test')
    parser.add_argument('--test_data_shrink_frac', type=float, default=0.15, help='shrink dataset to this fraction')
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
    parser.add_argument('--noise_beta_dc', type=float, default=0.1)
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

    parser.add_argument('--mapper_enable', type=str2bool, default=False)
    parser.add_argument('--mapper_train_frac', type=float, default=1.0)
    parser.add_argument('--mapper_path', type=str, default=mapper_path)
    parser.add_argument('--mapper_dataset', type=str,
                        default='')
    parser.add_argument('--mapper_target', type=str,
                        default='')

    parser.add_argument('--reducer_enable', type=str2bool, default=False)
    parser.add_argument('--reducer_train_frac', type=float, default=1.0)
    parser.add_argument('--reducer_train_label', type=str, default='train')
    parser.add_argument('--reducer_dataset', type=str,
                        default='')
    parser.add_argument('--reducer_layer', type=int,
                        default=-1)
    parser.add_argument('--reducer_alpha', type=int,
                        default=512)
    # # sip
    # parser.add_argument('--sip_enable', type=str2bool, default=True)
    # parser.add_argument('--sip_path', type=str,
    #                     default=attacker_path,
    #                     help='trained attacker model for b2tr')
    # parser.add_argument('--sip_model', type=str, default='gru', help='lstm, gru, linear')
    # parser.add_argument('--sip_dataset', type=str,
    #                     default='')
    # parser.add_argument('--sip_train_frac', type=float, default=1.0)
    # parser.add_argument('--sip_prefix', type=str, default='normal')
    # parser.add_argument('--sip_b2tr_enable', type=str2bool, default=True)
    # parser.add_argument('--sip_b2tr_layer', type=int, default=-1)
    # parser.add_argument('--sip_b2tr_target_layer', type=int, default=-1)
    # parser.add_argument('--sip_tr2t_enable', type=str2bool, default=True)
    # parser.add_argument('--sip_tr2t_target_layer', type=int, default=-1)
    # parser.add_argument('--sip_tr2t_layer', type=int, default=-1)
    # # GMA
    # parser.add_argument('--gma_enable', type=str2bool, default=True)
    # parser.add_argument('--gma_epochs', type=int, default=30)
    # parser.add_argument('--gma_beta', type=float, default=0.85)
    # parser.add_argument('--gma_lr', type=float, default=0.09)
    # parser.add_argument('--gma_init_temp', type=float, default=1.0)
    # # TAG
    # parser.add_argument('--tag_enable', type=str2bool, default=False)
    # parser.add_argument('--tag_epochs', type=int, default=400)
    # parser.add_argument('--tag_lr', type=float, default=0.09)
    # parser.add_argument('--tag_beta', type=float, default=0.85)
    # # LAMP
    # parser.add_argument('--lamp_freq', type=int, default=30)
    # # SMA
    # parser.add_argument('--sma_enable', type=str2bool, default=False)
    # parser.add_argument('--sma_epochs', type=int, default=100)
    # parser.add_argument('--sma_lr', type=float, default=0.001)
    # parser.add_argument('--sma_wd', type=float, default=0.01)
    # parser.add_argument('--sma_at', type=str, default='b2tr')
    # # GSMA
    # parser.add_argument('--gsma_enable', type=str2bool, default=False)
    # parser.add_argument('--gsma_epochs', type=int, default=100)
    # parser.add_argument('--gsma_lr', type=float, default=0.001)
    # parser.add_argument('--gsma_at', type=str, default='b2tr')
    # # EIA
    # parser.add_argument('--eia_enable', type=str2bool, default=False)
    # parser.add_argument('--eia_epochs', type=int, default=500)
    # parser.add_argument('--eia_lr', type=float, default=0.09)
    # parser.add_argument('--eia_temp', type=float, default=0.1)
    # parser.add_argument('--eia_wd', type=float, default=0.01)
    # parser.add_argument('--eia_mapped_to', type=int, default=1)
    # parser.add_argument('--eia_at', type=str, default='b2tr')
    #

    #
    # # ALT
    # parser.add_argument('--alt_enable', type=str2bool, default=False)
    # parser.add_argument('--alt_steps', type=int, default=3)
    # parser.add_argument('--alt_fwd_steps', type=int, default=64)
    # parser.add_argument('--alt_bwd_steps', type=int, default=18)


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
                      noise_beta_dc=args.noise_beta_dc,
                      collect_intermediates=True,
                      collect_all_layers=args.collect_all_layers,
                      dataset_type=args.dataset_label,
                      batch_size=args.batch_size
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
    if 'llama' in model_name or 'chatglm' in model_name or 'vicuna' in model_name:
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


def get_model(model_name='gpt2', task='lm', num_labels=2, tokenizer=None, load_bits=8, force_on_best_gpu=True,
              **kwargs):
    clz = GPT2SplitLMHeadModel
    if model_name.startswith('gpt2'):
        if task == 'lm':
            clz = GPT2SplitLMHeadModel
        elif task == 'clsf':
            clz = GPT2SplitClassificationModel
    elif model_name.startswith('bert'):
        clz = BertForSequenceClassificationSplitModel
    elif model_name.startswith('roberta'):
        clz = RobertaForSequenceClassificationSplitModel
    elif 't5' in model_name or 'ul2' in model_name:
        clz = T5ForConditionalGenerationSplitModel
    elif 'chatglm' in model_name:
        clz = ChatGLMForConditionalGenerationSplit
    elif 'llama' in model_name or 'vicuna' in model_name:
        clz = LLAMA2SplitLMHeadModel
    elif 'wizard' in model_name:
        clz = WizardForCausalLMSplit
    if 'llama' in model_name or 'chatglm' in model_name or 'vicuna' in model_name or 'wizard' in model_name:
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
    model = clz.from_pretrained(get_model_path(model_name), device_map=device_map, **kwargs)
    if tokenizer is not None and 'chatglm' not in model_name:
        if model.config.pad_token_id is not None:
            tokenizer.pad_token_id = model.config.pad_token_id
        if model.config.eos_token_id is not None:
            tokenizer.pad_token_id = model.config.eos_token_id
    return model


def get_model_and_tokenizer(model_name='gpt2', task='lm', num_labels=2, force_on_best_gpu=True, **kwargs):
    if model_name.startswith('vit'):
        processor = ViTImageProcessor.from_pretrained(
            os.path.join(model_download_dir, f'google/{model_name}-patch16-224'))
        model = ViTForImageClassificationSplit.from_pretrained(
            os.path.join(model_download_dir, f'google/{model_name}-patch16-224'))
        return model, processor
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, task, num_labels, tokenizer, force_on_best_gpu=force_on_best_gpu, **kwargs)

    return model, tokenizer


def get_dataset(dataset_name, tokenizer, client_ids=None, shrink_frac=1.0) -> FedDataset:
    if client_ids is None:
        client_ids = []
    if ',' in dataset_name:
        dataset_names = dataset_name.split(',')
        dataset_classes = [get_dataset_class(dn) for dn in dataset_names]
        return MixtureFedDataset(tokenizer, client_ids, shrink_frac, dataset_names, dataset_classes)
    else:
        return get_dataset_class(dataset_name)(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=shrink_frac)


def get_dataset_class(dataset_name):
    if dataset_name == 'piqa':
        dataset_cls = PIQAFedDataset
    elif dataset_name == 'piqa-mini':
        dataset_cls = PIQAMiniFedDataset
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
    elif dataset_name == 'sensimarked':
        dataset_cls = SensiMarkedFedDataset
    elif dataset_name == 'sensireplaced':
        dataset_cls = SensiReplacedFedDataset
    elif dataset_name == 'sensimasked':
        dataset_cls = SensiMaskedFedDataset
    elif dataset_name == 'hc3cn':
        dataset_cls = HC3CNFedDataset
    elif dataset_name == 'imagewoof':
        dataset_cls = ImageWoofFedDataset
    else:
        raise AttributeError
    return dataset_cls

#
# def get_fsha_attacker(dra_config: SIPAttackerArguments):
#     dataset = dra_config.dataset
#     if dataset is None:
#         dataset = dra_config.target_dataset
#     model_name = dra_config.target_model_name
#     if model_name == 'llama2':
#         model_name += f"-{dra_config.target_model_load_bits}bits"
#     attacker_path = fsha_path + f'{model_name}/{dataset}/'
#     match = False
#     for d in os.listdir(attacker_path):
#         pattern = f'{DRA_train_label[dra_config.dataset]}*{dra_config.train_frac:.3f}'
#         if ',' in dra_config.dataset:
#             pattern = f'Tr{dra_config.train_frac:.3f}'
#         if d.startswith(pattern):
#             attacker_path = os.path.join(attacker_path, d) + '/'
#             match = True
#             break
#     assert match
#     attacker_path_1 = None
#     if dra_config.b2tr_enable:
#         sp1 = int(dra_config.target_sps.split('-')[0])
#         if dra_config.b2tr_sp >= 0:
#             sp1 = dra_config.b2tr_sp
#         attacker_path_1 = attacker_path + f'/b2tr-{sp1}/'
#         l = sorted(list(os.listdir(attacker_path_1)), key=lambda x: float(x.split('_')[-1]))[-1]
#         attacker_path_1 = os.path.join(attacker_path_1, l)
#
#     attacker = FSHAAttacker.from_pretrained(attacker_path_1)
#
#     return attacker
