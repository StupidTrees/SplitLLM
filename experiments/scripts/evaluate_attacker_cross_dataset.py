import argparse
import os
import sys

import torch
import wandb
from transformers import AutoTokenizer

sys.path.append(os.path.abspath('../..'))
from experiments.scripts.basic_strategy import BaseSFLStrategy
from sfl.utils.experiments import add_sfl_params
from sfl.utils.model import calculate_rouge
import sfl
from sfl.simulator.simulator import SFLSimulator
from sfl.config import FLConfig, Intermediate
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.utils.training import set_random_seed, get_dataset_class, get_attacker_class, extract_attacker_path


# 定义Client本地学习策略
class QAFLStrategy(BaseSFLStrategy):

    def normal_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[int, Intermediate],
                                  batch, logs):
        rg = [False]
        if args.attacker_search:
            rg.append(True)
        for search in rg:
            with torch.no_grad():
                for idx, inter in all_inter.items():
                    suffix = inter.type
                    self.attacker1.to(self.simulator.device)
                    if search:
                        attacked = self.attacker1.search(inter.fx, self.llm)
                    else:
                        attacked = self.attacker1(inter.fx.to(self.attacker1.device))
                    rouge_res = calculate_rouge(self.tokenizer, attacked, batch['input_text'])
                    self.log_to_all_result(client_id, f'attacker_{idx}_{suffix}', rouge_res['rouge-l']['f'])
                    logs[f'attacker_{idx}_{suffix}_step'] = rouge_res['rouge-l']['f']


def sfl_with_attacker(args):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    model = GPT2SplitLMHeadModel.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 50256
    # 加载攻击模型
    attacker1, _ = extract_attacker_path(args, get_attacker_class(args.attacker_model))

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
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
                      dataset_type=args.dataset_label
                      )
    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    test_dataset = dataset_cls(tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=QAFLStrategy(args, tokenizer, attacker1, None, model, test_loader),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config)
    wandb.init(
        project=args.exp_name,
        name=f"{args.dataset}.{args.dataset_label}-attacked-by-{args.attacker_dataset}.{args.attacker_train_label}-on-{args.attacker_b2tr_sp}",
        config=args
    )
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
