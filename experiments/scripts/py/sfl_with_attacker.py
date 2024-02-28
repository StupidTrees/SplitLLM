import argparse
import os
import sys

import wandb

sys.path.append(os.path.abspath('../../..'))


from sfl.utils.model import set_random_seed
from sfl.simulator.simulator import SFLSimulator
from sfl.simulator.strategy import BaseSFLStrategy
from sfl.utils.exp import get_model_and_tokenizer, get_dra_config, get_dra_attacker, get_fl_config, get_dlg_attacker, \
    get_dataset_class, add_sfl_params


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name, args.task_type)
    # 加载攻击模型
    attacker1, attacker2 = get_dra_attacker(get_dra_config(args))
    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    # 加载TAG攻击模型
    dlg = get_dlg_attacker(args)
    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    test_dataset = dataset_cls(tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=BaseSFLStrategy(args, model, tokenizer, test_loader, attacker1, attacker2, dlg),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config)
    wandb.init(
        project=args.exp_name,
        name=f"{args.split_points}",
        config=args
    )
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
