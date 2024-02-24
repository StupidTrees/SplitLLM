import argparse
import os
import sys

import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, AdamW

sys.path.append(os.path.abspath('../..'))
from sfl.utils.experiments import add_sfl_params
from sfl.utils.model import calculate_rouge_text
import sfl
from sfl.config import FLConfig
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.utils.training import set_random_seed, get_dataset_class, get_attacker_class, extract_attacker_path, \
    get_best_gpu


def pre_ft(llm, data_loader):
    llm.train()
    # 不收集中间结果
    bk_ci = llm.fl_config.collect_intermediates
    bk_th = llm.fl_config.trigger_hook
    bk_nm = llm.fl_config.noise_mode
    llm.fl_config.collect_intermediates = False
    llm.fl_config.trigger_hook = False
    llm.fl_config.noise_mode = 'none'
    # 微调bottom和top
    tune = [p for p in llm.parameters() if p.requires_grad]
    optimizer = AdamW(tune, lr=1e-5)
    with tqdm(total=len(data_loader)) as pbar:
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(llm.device)
            attention_mask = batch['input_att_mask'].to(llm.device)
            outputs = llm(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            pbar.set_description(f'Pre-FT Loss {loss.item():.3f}')
            loss.backward()
            optimizer.step()
            pbar.update(1)
    llm.fl_config.collect_intermediates = bk_ci
    llm.fl_config.trigger_hook = bk_th
    llm.fl_config.noise_mode = bk_nm
    llm.eval()


def sfl_with_attacker(args):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    model = GPT2SplitLMHeadModel.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 50256
    # 加载攻击模型
    attacker, attacker2 = extract_attacker_path(args, get_attacker_class(args.attacker_model))
    attacked_samples = []

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = FLConfig(split_point_1=int(args.split_points.split('-')[0]),
                      split_point_2=int(args.split_points.split('-')[1]),
                      use_lora_at_trunk=args.lora_at_trunk,  # 在trunk部分使用LoRA
                      use_lora_at_top=args.lora_at_top,
                      use_lora_at_bottom=args.lora_at_bottom,
                      top_and_bottom_from_scratch=args.client_from_scratch,  # top和bottom都不采用预训练参数.
                      noise_mode=args.noise_mode,
                      noise_scale=args.noise_scale,  # 噪声大小
                      collect_intermediates=True,
                      dataset_type=args.dataset_label,
                      collect_all_layers=args.collect_all_layers,
                      trigger_hook=True
                      )

    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    device = get_best_gpu()

    def hook(hidden, input_ids, layer_past):
        with torch.no_grad():
            attacked = attacker(hidden.to(attacker.device))
            attacked_samples.append(attacked)

    model.config_sfl(config, None, [hook])
    model = model.convert_to_lora_model()
    wandb.init(
        project=args.exp_name,
        name=f"C-Dxp{args.noise_scale}-{args.dataset}.{args.dataset_label}-Pre{args.pre_ft_dataset}.{args.pre_ft_data_label}.{args.pre_ft_data_shrink_frac:.2f}",
        config=args
    )
    attacker.to(device)
    model.to(device)
    # 加载Pre-FT数据集
    if args.pre_ft_dataset is not None and len(args.pre_ft_dataset) > 0:
        pre_ft_dataset = get_dataset_class(args.pre_ft_dataset)(tokenizer=tokenizer, client_ids=[])
        pre_ft_loader = pre_ft_dataset.get_dataloader_unsliced(args.batch_size, args.pre_ft_data_label,
                                                               shrink_frac=args.pre_ft_data_shrink_frac)
        pre_ft(model, pre_ft_loader)
    sum_rouge_first = 0
    sum_rouge_merged = 0
    sample_num = 0
    loader = fed_dataset.get_dataloader_unsliced(1, args.dataset_label, shrink_frac=args.data_shrink_frac)
    with tqdm(total=len(loader)) as pbar:
        for batch in loader:
            attacked_samples.clear()
            raw_text = batch['input_text'][0]
            # print(f"Q: {raw_text}")
            res = model.generate(tokenizer(raw_text, return_tensors='pt').input_ids.to(model.device),
                                 max_new_tokens=10,
                                 generation_config=GenerationConfig(use_cache=False,
                                                                    pad_token_id=tokenizer.eos_token_id))
            min_len = min([t.shape[1] for t in attacked_samples])
            for i in range(len(attacked_samples)):
                attacked_samples[i] = attacked_samples[i][:, :min_len, :]
            merged_probs = torch.mean(torch.stack(attacked_samples), dim=0)
            txt_first = tokenizer.decode(attacked_samples[0].argmax(dim=-1)[0], skip_special_tokens=True)
            txt_merged = tokenizer.decode(merged_probs.argmax(dim=-1)[0], skip_special_tokens=True)
            rouge_first = calculate_rouge_text([txt_first], [raw_text])
            rouge_merged = calculate_rouge_text([txt_merged], [raw_text])
            sum_rouge_merged += rouge_merged['rouge-l']['f']
            sum_rouge_first += rouge_first['rouge-l']['f']
            sample_num += 1
            pbar.set_description(
                f"rouge_first: {sum_rouge_first / sample_num:.4f}, rouge_merged: {sum_rouge_merged / sample_num:.4f}")
            pbar.update(1)
            attacked_samples.clear()
            wandb.log({'rouge_first': sum_rouge_first / sample_num,
                       'rouge_merged': sum_rouge_merged / sample_num})
            # print(rouge_first['rouge-l']['f'], rouge_merged['rouge-l']['f'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
