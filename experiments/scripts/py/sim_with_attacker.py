import os
import sys

import wandb

sys.path.append(os.path.abspath('../../..'))
from sfl.model.attacker.alt_attacker import ALTAttacker
from sfl.strategies.sl_strategy_with_attacker import SLStrategyWithAttacker
from sfl.utils.model import set_random_seed
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *
from sfl.model.attacker.dlg_attacker import TAGAttacker, LAMPAttacker
from sfl.model.attacker.eia_attacker import SmashedDataMatchingAttacker, EmbeddingInversionAttacker
from sfl.model.attacker.sip_attacker import SIPAttacker


def sfl_with_attacker(args, unkown_args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # 配置切分学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    # 如果有DimReducer,配置
    dim_reducer = None
    reducer_args = get_reducer_args()
    if reducer_args.enable:
        config.reducer_enable = True
        dim_reducer = get_dim_reducer(args, reducer_args)
    # 配置FL
    model.config_sfl(config, dim_reducer=dim_reducer)
    model.train()

    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, args.test_data_label,
                                                       shrink_frac=args.test_data_shrink_frac,
                                                       max_seq_len=args.dataset_max_seq_len)
    sample_batch = next(iter(fed_dataset.get_dataloader_unsliced(3, 'train')))
    # 配置攻击者
    # Name | AttackerObj | ConfigPrefix | Initialized From
    attackers_conf = [('SIP', SIPAttacker(), 'sip', None),
                      ('ALT', ALTAttacker(config, model), 'alt', 'SIP_b2tr'),
                      ('BiSR(b)', TAGAttacker(config, model), 'gma', f'SIP_b2tr'),
                      ('BiSR(f)', SmashedDataMatchingAttacker(), 'sma', f'SIP_b2tr'),
                      ('BiSR(b+f)', SmashedDataMatchingAttacker(), 'gsma', 'BiSR(b)'),
                      ('EIA', EmbeddingInversionAttacker(), 'eia', None),
                      ('TAG', TAGAttacker(config, model), 'tag', None),
                      ('LAMP', LAMPAttacker(config, model), 'lamp', None),
                      ]
    # 配置联邦切分学习策略和模拟器
    strategy = SLStrategyWithAttacker(args, config, model, tokenizer,
                                      test_loader=test_loader,
                                      sample_batch=sample_batch,
                                      attackers_conf=attackers_conf)
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

    args_dict = vars(args)
    args_dict.update(unkown_args)
    # 配置wandb
    wandb.init(
        project=args.exp_name,
        name=args.case_name,
        config=args_dict
    )
    # 开始模拟
    model.train()
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args, remain = parser.parse_known_args()
    remain_dict = args_to_dict(remain)
    set_random_seed(args.seed)
    sfl_with_attacker(args, remain_dict)
