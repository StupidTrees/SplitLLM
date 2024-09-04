import argparse
import os
import sys

import wandb

sys.path.append(os.path.abspath('../../..'))
from sfl.utils.exp import get_model_and_tokenizer, get_fl_config, get_reducer_args, get_dim_reducer, get_dataset, \
    add_sfl_params, args_to_dict
from sfl.model.attacker.sma_attacker import SmashedDataMatchingAttacker
from sfl.model.attacker.ltn_attacker import ALTAttacker
from sfl.strategies.sl_strategy_with_attacker import SLStrategyWithAttacker
from sfl.utils.model import set_random_seed
from sfl.simulator.simulator import SFLSimulator
from sfl.model.attacker.dlg_attacker import TAGAttacker, LAMPAttacker
from sfl.model.attacker.eia_attacker import EmbeddingInversionAttacker
from sfl.model.attacker.sip_attacker import SIPAttacker


def sfl_with_attacker(args, unkown_args):
    model, tokenizer = get_model_and_tokenizer(args.model_name, load_bits=args.load_bits)

    # Config SL
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    # Config DimReducer if needed
    dim_reducer = None
    reducer_args = get_reducer_args()
    if reducer_args.enable:
        config.reducer_enable = True
        dim_reducer = get_dim_reducer(args, reducer_args)
    model.config_sfl(config, dim_reducer=dim_reducer)
    model.train()

    # Load dataset
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac, completion_only=args.completion_only)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[], completion_only=args.completion_only)
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, args.test_data_label,
                                                       shrink_frac=args.test_data_shrink_frac,
                                                       max_seq_len=args.dataset_max_seq_len)
    sample_batch = next(iter(fed_dataset.get_dataloader_unsliced(3, 'train')))
    # Set up attackers
    # Name | AttackerObj | ConfigPrefix | Initialized From
    attackers_conf = [('SIP', SIPAttacker(), 'sip', None),
                      ('ALT', ALTAttacker(config, model), 'alt', 'SIP_b2tr'),
                      ('BiSR(b)', TAGAttacker(config, model), 'gma', f'SIP_b2tr'),
                      ('BiSR(f)', SmashedDataMatchingAttacker(), 'sma', f'SIP_b2tr'),
                      ('BiSR(b+f)', SmashedDataMatchingAttacker(), 'gsma', 'BiSR(b)'),
                      ('EIA', EmbeddingInversionAttacker(), 'eia', None),
                      ('BiSR(f2)', EmbeddingInversionAttacker(), 'xma', f'SIP_b2tr'),
                      ('TAG', TAGAttacker(config, model), 'tag', None),
                      ('LAMP', LAMPAttacker(config, model), 'lamp', None),
                      ]
    # Initialize strategy
    strategy = SLStrategyWithAttacker(args, config, model, tokenizer,
                                      test_loader=test_loader,
                                      sample_batch=sample_batch,
                                      attackers_conf=attackers_conf)
    # Initialize simulator
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=strategy,
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)

    args_dict = vars(args)
    args_dict.update(unkown_args)
    # Config wandb
    wandb.init(
        project=args.exp_name,
        name=args.case_name,
        config=args_dict
    )

    # Run pre-fine-tuning, if needed
    if args.pre_ft_dataset is not None and len(args.pre_ft_dataset) > 0:
        pre_ft_dataset = get_dataset(args.pre_ft_dataset, tokenizer=tokenizer, client_ids=[])
        pre_ft_loader = pre_ft_dataset.get_dataloader_unsliced(args.batch_size, args.pre_ft_data_label)
        simulator.pre_ft(pre_ft_loader, ['bottom', 'top'], max_steps=args.pre_ft_max_steps)

    # Simulator run
    model.train()
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args, remain = parser.parse_known_args()
    remain_dict = args_to_dict(remain)
    set_random_seed(args.seed)
    sfl_with_attacker(args, remain_dict)
