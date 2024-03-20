import os
import sys

import torch
import wandb

sys.path.append(os.path.abspath('../../..'))

from sfl.simulator.strategy import BaseSFLStrategy
from sfl.utils.model import calculate_rouge, Intermediate, set_random_seed
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *


# 定义Client本地学习策略
class MultiLayerDRAFLStrategy(BaseSFLStrategy):
    """
    每一轮触发攻击：攻击每一层的中间输出
    """

    def normal_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[int, Intermediate],
                                  batch, logs):
        with torch.no_grad():
            for idx, inter in all_inter.items():
                # suffix = inter.type
                self.dra1.to(self.simulator.device)
                attacked = self.dra1(inter.fx.to(self.dra1.device))
                rouge_res = calculate_rouge(self.tokenizer, attacked, batch['input_text'])
                self.log_to_all_result(client_id, f'DRA_{idx}_RLF', rouge_res['rouge-l']['f'])
                self.log_to_all_result(client_id, f'DRA_{idx}_R1F', rouge_res['rouge-1']['f'])
                self.log_to_all_result(client_id, f'DRA_{idx}_R2F', rouge_res['rouge-2']['f'])
                logs[f'DRA_{idx}_step_RLF'] = rouge_res['rouge-l']['f']


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
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer)
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=MultiLayerDRAFLStrategy(args, model, tokenizer, test_loader, attacker1, None),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config)
    wandb.init(
        project=args.exp_name,
        name=f"{args.dataset}-{args.dataset_label}-trained-on-{args.attacker_b2tr_sp}",
        config=args
    )
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
