import argparse
import os
import sys

import wandb

sys.path.append(os.path.abspath('../../..'))
from experiments.scripts.py.evaluate_dra_cross_layer import MultiLayerDRAFLStrategy
from sfl.utils.model import set_random_seed

from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import get_model_and_tokenizer, get_fl_config, add_sfl_params, get_dataset, get_attacker_class, \
    get_dra_attacker, get_dra_config


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # 加载攻击模型
    attacker1, _ = get_dra_attacker(get_dra_config(args))
    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=MultiLayerDRAFLStrategy(args,
                                                              model, tokenizer, test_loader, attacker1),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)
    wandb.init(
        project=args.exp_name,
        name=args.case_name,
        config=args,
    )
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
