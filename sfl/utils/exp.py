import argparse
import os

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

import sfl
from sfl.config import FLConfig, attacker_path, DRAConfig
from sfl.model.attacker.dlg_attacker import GPT2TopDLGAttacker, T5DecoderDLGAttacker, LLAMA2TopDLGAttacker
from sfl.model.attacker.dra_attacker import LSTMDRAttacker, GRUDRAttacker, LinearDRAttacker, \
    TransformerEncoderDRAttacker
from sfl.model.llm.bert.bert_wrapper import BertForSequenceClassificationSplitModel
from sfl.model.llm.gpt2.gpt2_wrapper import GPT2SplitLMHeadModel, GPT2SplitClassificationModel
from sfl.model.llm.llama2.llama2_wrapper import LLAMA2SplitLMHeadModel
from sfl.model.llm.roberta.roberta_wrapper import RobertaForSequenceClassificationSplitModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.t5.t5wrapper import T5ForConditionalGenerationSplitModel
from sfl.simulator.dataset import CodeAlpacaFedDataset, DialogSumFedDataset, IMDBFedDataset, PIQAFedDataset, \
    GSM8KFedDataset, WikiTextFedDataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_sfl_params(parser):
    parser.add_argument('--exp_name', type=str, default='compare_tag')
    parser.add_argument('--model_name', type=str, default='gpt2-large')
    parser.add_argument('--pre_ft_dataset', type=str, default='')
    parser.add_argument('--pre_ft_data_label', type=str, default='train')
    parser.add_argument('--pre_ft_data_shrink_frac', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, default='wikitext')
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
    parser.add_argument('--noise_mode', type=str, default='none')
    parser.add_argument('--task_type', type=str, default='lm')
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--attacker_model', type=str, default='gru', help='lstm, gru, linear')
    parser.add_argument('--attacker_train_label', type=str, default='validation')
    parser.add_argument('--attacker_train_frac', type=float, default=1.0)
    parser.add_argument('--attacker_prefix', type=str, default='normal')
    parser.add_argument('--attacker_search', type=str2bool, default=False)
    parser.add_argument('--attacker_freq', type=int, default=25, help='attack every * steps')
    parser.add_argument('--attacker_samples', type=int, default=10, help='attack how many batches each time')
    parser.add_argument('--attacker_dataset', type=str,
                        default=None)
    parser.add_argument('--attacker_path', type=str,
                        default=attacker_path,
                        help='trained attacker model for b2tr')
    parser.add_argument('--attacker_b2tr_enable', type=str2bool, default=True)
    parser.add_argument('--attacker_b2tr_sp', type=int, default=15)
    parser.add_argument('--attacker_tr2t_enable', type=str2bool, default=True)
    parser.add_argument('--attacker_tr2t_sp', type=int, default=15)
    parser.add_argument('--client_num', type=int, default=1)
    parser.add_argument('--global_round', type=int, default=4)
    parser.add_argument('--client_from_scratch', type=str2bool, default=False)
    parser.add_argument('--dlg_enable', type=str2bool, default=True)
    parser.add_argument('--dlg_epochs', type=int, default=30)
    parser.add_argument('--dlg_adjust', type=int, default=0)
    parser.add_argument('--dlg_beta', type=float, default=0.9)
    parser.add_argument('--dlg_init_with_dra', type=str2bool, default=True,
                        help='initialize GT vector with DRA attacker')
    parser.add_argument('--dlg_dra_reg', type=float, default=0.0,
                        help='Add regularization term to make GT closer to DRA result')
    parser.add_argument('--dlg_temp_range', type=float, default=0.0)
    parser.add_argument('--dlg_further_ft', type=int, default=0)
    parser.add_argument('--self_pt_enable', type=str2bool, default=False)
    parser.add_argument('--client_steps', type=int, default=50)
    parser.add_argument('--client_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_to_wandb', type=str2bool, default=True)


def get_fl_config(args) -> FLConfig:
    config = FLConfig(global_round=args.global_round,
                      client_evaluate_freq=args.evaluate_freq,
                      client_steps=args.client_steps,
                      client_epoch=args.client_epoch,  # 每轮联邦每个Client训x轮
                      split_point_1=int(args.split_points.split('-')[0]),
                      split_point_2=int(args.split_points.split('-')[1]),
                      use_lora_at_trunk=args.lora_at_trunk,  # 在trunk部分使用LoRA
                      use_lora_at_top=args.lora_at_top,
                      use_lora_at_bottom=args.lora_at_bottom,
                      top_and_bottom_from_scratch=args.client_from_scratch,  # top和bottom都不采用预训练参数.
                      noise_mode=args.noise_mode,
                      noise_scale=args.noise_scale,  # 噪声大小
                      collect_intermediates=True,
                      collect_all_layers=args.collect_all_layers,
                      dataset_type=args.dataset_label,
                      batch_size=args.batch_size
                      )
    return config


def get_dra_config(args) -> DRAConfig:
    res = DRAConfig()
    res.path = args.attacker_path
    res.model = args.attacker_model
    res.train_label = args.attacker_train_label
    res.train_frac = args.attacker_train_frac
    res.prefix = args.attacker_prefix
    res.dataset = args.attacker_dataset
    res.b2tr_sp = args.attacker_b2tr_sp
    res.b2tr_enable = args.attacker_b2tr_enable
    res.tr2t_sp = args.attacker_tr2t_sp
    res.tr2t_enable = args.attacker_tr2t_enable
    res.target_dataset = args.dataset
    res.target_model_name = args.model_name
    res.target_sps = args.split_points
    return res


def get_model_path(model_name):
    path = ''
    if model_name.startswith('gpt2'):
        path = os.path.join(sfl.config.model_download_dir, model_name)
    elif model_name == 'bert':
        path = os.path.join(sfl.config.model_download_dir, "google-bert/bert-base-uncased/")
    elif model_name.startswith('roberta'):
        path = os.path.join(sfl.config.model_download_dir, f"FacebookAI/{model_name}/")
    elif model_name.startswith('flan-t5'):
        path = os.path.join(sfl.config.model_download_dir, f"google/{model_name}")
    elif model_name.startswith('llama2'):
        path = os.path.join(sfl.config.model_download_dir, f"daryl149/llama-2-7b-chat-hf")
    return path


def get_tokenizer(model_name='gpt2'):
    return AutoTokenizer.from_pretrained(get_model_path(model_name))


def get_model(model_name='gpt2', task='lm', num_labels=2, tokenizer=None, **kwargs):
    clz = GPT2SplitLMHeadModel
    if model_name.startswith('gpt2'):
        if task == 'lm':
            clz = GPT2SplitLMHeadModel
        elif task == 'clsf':
            clz = GPT2SplitClassificationModel
    elif model_name == 'bert':
        clz = BertForSequenceClassificationSplitModel
    elif model_name.startswith('roberta'):
        clz = RobertaForSequenceClassificationSplitModel
    elif 't5' in model_name:
        clz = T5ForConditionalGenerationSplitModel
    elif 'llama2' in model_name:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # load_in_4bit=True,  # load the model into memory using 4-bit precision
            bnb_4bit_use_double_quant=True,  # use double quantition
            bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,  # use double quantition
            bnb_8bit_quant_type="nf8",  # use NormalFloat quantition
            bnb_8bit_compute_dtype=torch.bfloat16  # use hf for computing when we need

        )
        kwargs['quantization_config'] = bnb_config
        clz = LLAMA2SplitLMHeadModel

    if task == 'clsf':
        kwargs['num_labels'] = num_labels
    model = clz.from_pretrained(get_model_path(model_name), **kwargs)
    if tokenizer is not None:
        if model.config.pad_token_id is not None:
            tokenizer.pad_token_id = model.config.pad_token_id
        if model.config.eos_token_id is not None:
            tokenizer.pad_token_id = model.config.eos_token_id
    return model


def get_model_and_tokenizer(model_name='gpt2', task='lm', num_labels=2, **kwargs):
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, task, num_labels, tokenizer, **kwargs)

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
    attacker_cls = LSTMDRAttacker
    if attack_model == 'lstm':
        attacker_cls = LSTMDRAttacker
    elif attack_model == 'gru':
        attacker_cls = GRUDRAttacker
    elif attack_model == 'linear':
        attacker_cls = LinearDRAttacker
    elif attack_model == 'trans-enc':
        attacker_cls = TransformerEncoderDRAttacker
    return attacker_cls


def get_dra_attacker(dra_config: DRAConfig):
    dataset = dra_config.dataset
    if dataset is None:
        dataset = dra_config.target_dataset
    attacker_path = dra_config.path + f'{dra_config.target_model_name}/{dataset}/'
    match = False
    for d in os.listdir(attacker_path):
        if d.startswith(f'{dra_config.train_label}*{dra_config.train_frac:.3f}'):
            attacker_path = os.path.join(attacker_path, d) + '/'
            match = True
            break
    assert match
    attacker_path += f'{dra_config.model}'
    attacker_path_1 = None
    if dra_config.b2tr_enable:
        sp1 = int(dra_config.target_sps.split('-')[0])
        if dra_config.b2tr_sp >= 0:
            sp1 = dra_config.b2tr_sp
        attacker_path_1 = attacker_path + f'/b2tr-{sp1}/' + dra_config.prefix
        l = sorted(list(os.listdir(attacker_path_1)), key=lambda x: float(x.split('_')[-1]))[-1]
        attacker_path_1 = os.path.join(attacker_path_1, l)
    attacker_path_2 = None
    if dra_config.tr2t_enable:
        sp2 = int(dra_config.target_sps.split('-')[1])
        if dra_config.tr2t_sp >= 0:
            sp2 = dra_config.tr2t_sp
        attacker_path_2 = attacker_path + f'/tr2t-{sp2}/' + dra_config.prefix
        if not os.path.exists(attacker_path_2):
            attacker_path_2 = attacker_path_2.replace('tr2t', 'b2tr')
        l = sorted(list(os.listdir(attacker_path_2)), key=lambda x: float(x.split('_')[-1]))[-1]
        attacker_path_2 = os.path.join(attacker_path_2, l)
    attacker, attacker2 = None, None
    if attacker_path_1:
        attacker = get_attacker_class(dra_config.model).from_pretrained(attacker_path_1)
    if attacker_path_2:
        attacker2 = get_attacker_class(dra_config.model).from_pretrained(attacker_path_2)
    return attacker, attacker2


def get_dlg_attacker(llm: SplitWrapperModel):
    mocker = None
    assert llm.fl_config is not None
    # !需要在LoRA加上去之前进行复制
    if isinstance(llm, GPT2SplitLMHeadModel):
        mocker = GPT2TopDLGAttacker(llm.fl_config, llm)
    elif isinstance(llm, LLAMA2SplitLMHeadModel):
        mocker = LLAMA2TopDLGAttacker(llm.fl_config, llm)
        # print(llm.config)
    elif isinstance(llm, T5ForConditionalGenerationSplitModel):
        mocker = T5DecoderDLGAttacker(llm.fl_config, llm)

    return mocker
