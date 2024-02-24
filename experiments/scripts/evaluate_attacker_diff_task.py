import argparse
import os
import sys

import wandb

sys.path.append(os.path.abspath('../..'))

from experiments.scripts.basic_strategy import BaseSFLStrategy
from sfl.utils.experiments import add_sfl_params
from sfl.simulator.simulator import SFLSimulator
from sfl.config import FLConfig, Intermediate
from sfl.utils.training import set_random_seed, get_dataset_class, get_attacker_class, extract_attacker_path, \
    get_tokenizer, get_model


# 定义Client本地学习策略
class QAFLStrategy(BaseSFLStrategy):
    def normal_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[int, Intermediate],
                                  batch, logs):
        pass


def sfl_with_attacker(args):
    # 加载tokenizer
    tokenizer = get_tokenizer(args.model_name)
    # 加载攻击模型
    attacker1, attacker2 = extract_attacker_path(args, get_attacker_class(args.attacker_model))

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
    test_loader = test_dataset.get_dataloader_unsliced(1, args.test_data_label, shrink_frac=args.test_data_shrink_frac)
    print('Test length:', len(test_loader))
    # 加载模型
    model = get_model(args.model_name, args.task_type, fed_dataset.num_labels, tokenizer)
    # if fed_dataset.task_type == 'clsf':
    #     model = GPT2SplitClassificationModel.from_pretrained(
    #         os.path.join(sfl.config.model_download_dir, args.model_name),
    #         num_labels=fed_dataset.num_labels, )
    # else:
    #     model = GPT2SplitLMHeadModel.from_pretrained(
    #         os.path.join(sfl.config.model_download_dir, args.model_name))
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=QAFLStrategy(args, tokenizer, attacker1, attacker2, model, test_loader),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)
    simulator.strategy.task_type = args.task_type

    wandb.init(
        project=args.exp_name,
        name=f"{args.model_name}-{args.dataset}.{args.dataset_label}-{args.task_type}",
        config=args
    )
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    parser.add_argument('--task_type', type=str, default=None)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
