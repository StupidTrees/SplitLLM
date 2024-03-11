import argparse
import os
import sys

import wandb

from sfl.utils.model import set_random_seed

sys.path.append(os.path.abspath('../../..'))

from sfl.simulator.strategy import BaseSFLStrategy
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import get_tokenizer, get_model, get_fl_config, \
    add_sfl_params, get_dra_attacker, get_dra_config, get_dataset


def sfl_with_attacker(args):
    # 加载tokenizer
    tokenizer = get_tokenizer(args.model_name)
    # 加载攻击模型
    attacker1, attacker2 = get_dra_attacker(get_dra_config(args))

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer)
    test_loader = test_dataset.get_dataloader_unsliced(1, args.test_data_label, shrink_frac=args.test_data_shrink_frac)

    # 加载模型
    model = get_model(args.model_name, args.task_type, fed_dataset.num_labels, tokenizer)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=BaseSFLStrategy(args, model, tokenizer, test_loader, attacker1, attacker2),
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
