import os
import sys

import wandb

sys.path.append(os.path.abspath('../../..'))
from sfl.strategies.sl_strategy_with_attacker import SLStrategyWithAttacker
from sfl.utils.model import set_random_seed
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)

    # 加载TAG攻击模型
    model.config_sfl(config, None)
    model.train()

    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, args.test_data_label,
                                                       shrink_frac=args.test_data_shrink_frac,
                                                       max_seq_len=args.dataset_max_seq_len)
    sample_batch = next(iter(fed_dataset.get_dataloader_unsliced(3, 'train')))
    strategy = SLStrategyWithAttacker(args, config, model, tokenizer,
                                      test_loader=test_loader,
                                      sample_batch=sample_batch)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=strategy,
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)
    # 加载Pre-FT数据集
    if args.pre_ft_dataset is not None and len(args.pre_ft_dataset) > 0:
        pre_ft_dataset = get_dataset(args.pre_ft_dataset, tokenizer=tokenizer, client_ids=[])
        pre_ft_loader = pre_ft_dataset.get_dataloader_unsliced(args.batch_size, args.pre_ft_data_label)
        simulator.pre_ft(pre_ft_loader, ['bottom', 'top'], max_steps=args.pre_ft_max_steps)

    wandb.init(
        project=args.exp_name,
        name=args.case_name,
        config=args
    )
    model.train()
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
