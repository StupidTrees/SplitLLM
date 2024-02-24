import argparse

from sfl import config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_sfl_params(parser):
    parser.add_argument('--exp_name', type=str, default='compare_tag')
    parser.add_argument('--model_name', type=str, default='gpt2-large')
    parser.add_argument('--pre_ft_dataset', type=str, default='')
    parser.add_argument('--pre_ft_data_label', type=str, default='train')
    parser.add_argument('--pre_ft_data_shrink_frac', type=float, default=0.2)

    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--dataset_label', type=str, default='train')
    parser.add_argument('--data_shrink_frac', type=float, default=0.15, help='shrink dataset to this fraction')
    parser.add_argument('--test_data_label', type=str, default='test')
    parser.add_argument('--test_data_shrink_frac', type=float, default=0.15, help='shrink dataset to this fraction')
    parser.add_argument('--evaluate_freq', type=int, default=25)
    parser.add_argument('--collect_all_layers', type=str2bool, default=False,
                        help='collect intermediates of all layers')
    parser.add_argument('--split_points', type=str, default='6-30', help='split points, b2tr-tr2t')
    parser.add_argument('--lora_at_trunk', type=str2bool, default=True, help='use LoRA at trunk part')
    parser.add_argument('--lora_at_bottom', type=str2bool, default=False, help='use LoRA at bottom part')
    parser.add_argument('--lora_at_top', type=str2bool, default=False, help='use LoRA at top part')
    parser.add_argument('--noise_mode', type=str, default='none')
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--attacker_model', type=str, default='gru', help='lstm, gru, linear')
    parser.add_argument('--attacker_train_label', type=str, default='validation')
    parser.add_argument('--attacker_train_frac', type=float, default=1.0)
    parser.add_argument('--attacker_prefix', type=str, default='normal')
    parser.add_argument('--attacker_search', type=str2bool, default=False)
    parser.add_argument('--attacker_freq', type=int, default=25, help='attack every * steps')
    parser.add_argument('--attacker_samples', type=int, default=10, help='attack how many batches each time')
    parser.add_argument('--attacker_dataset', type=str,
                        default='wikitext')
    parser.add_argument('--attacker_path', type=str,
                        default=config.attacker_path,
                        help='trained attacker model for b2tr')
    parser.add_argument('--attacker_b2tr_enable', type=str2bool, default=True)
    parser.add_argument('--attacker_b2tr_sp', type=int, default=15)
    parser.add_argument('--attacker_tr2t_enable', type=str2bool, default=True)
    parser.add_argument('--attacker_tr2t_sp', type=int, default=15)
    parser.add_argument('--client_num', type=int, default=1)
    parser.add_argument('--global_round', type=int, default=4)
    parser.add_argument('--client_from_scratch', type=str2bool, default=False)
    parser.add_argument('--dlg_enable', type=str2bool, default=True)
    parser.add_argument('--dlg_epochs', type=int, default=300)
    parser.add_argument('--dlg_adjust', type=int, default=0)
    parser.add_argument('--dlg_beta', type=float, default=0.8)
    parser.add_argument('--self_pt_enable', type=str2bool, default=False)
    parser.add_argument('--client_steps', type=int, default=50)
    parser.add_argument('--client_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
