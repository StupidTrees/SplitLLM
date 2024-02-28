import argparse
import os
import sys

import wandb


sys.path.append(os.path.abspath('../../..'))

from experiments.scripts.evaluate_dra_cross_layer import MultiLayerDRAFLStrategy
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import set_random_seed, get_dataset_class, get_attacker_class, extract_attacker_path, \
    get_model_and_tokenizer, get_fl_config, add_sfl_params



def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # 加载攻击模型
    attacker1, _ = extract_attacker_path(args, get_attacker_class(args.attacker_model))

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    test_dataset = dataset_cls(tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=MultiLayerDRAFLStrategy(args, tokenizer, attacker1, None, model, test_loader),
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
