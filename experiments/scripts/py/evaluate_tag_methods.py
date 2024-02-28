import os
import sys
from typing import Any

import torch
import wandb

sys.path.append(os.path.abspath('../../..'))


from sfl.simulator.strategy import BaseSFLStrategy

from sfl.utils.model import calculate_rouge, Intermediate, set_random_seed
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *


# 定义Client本地学习策略
class QAFLStrategy(BaseSFLStrategy):

    def sample_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[Any, Intermediate],
                                  batch, logs):
        encoder_inter = all_inter.get('encoder', None)
        with torch.no_grad():
            for type, atk in zip(['tr2t', 'b2tr'], [self.dra1, self.dra2]):
                if atk is None:
                    continue
                atk.to(self.simulator.device)
                inter = b2tr_inter if type == 'b2tr' else tr2t_inter
                if self.llm.type == 'encoder-decoder':
                    attacked = atk(torch.concat([encoder_inter.fx.to(
                        self.simulator.device), inter.fx.to(atk.device)], dim=1))
                else:
                    attacked = atk(inter.fx.to(atk.device))
                rouge_res = calculate_rouge(self.tokenizer, attacked, batch['input_text'])
                self.log_to_sample_result(client_id, f'attacker_{type}', rouge_res['rouge-l']['f'])
                self.log_to_all_result(client_id, f'attacker_{type}', rouge_res['rouge-l']['f'])
                logs[f'attacker_{type}_step'] = rouge_res['rouge-l']['f']
        gt_init = None
        if self.args.dlg_init_with_dra:
            gt_init = attacked
        gt = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device), tr2t_inter.grad.to(self.simulator.device),
                          epochs=self.args.dlg_epochs,
                          adjust=args.dlg_adjust,
                          beta=self.args.dlg_beta,
                          gt_init=gt_init,
                          gt_reg=self.args.dlg_dra_reg,
                          temp_range=self.args.dlg_temp_range,
                          further_ft=self.args.dlg_further_ft,
                          encoder_inter=None if encoder_inter is None else encoder_inter.fx.to(
                              self.simulator.device)
                          )
        if self.llm.type == 'encoder-decoder':
            # replace the latter half of attacked to gt
            attacked[:, -gt.shape[1]:, :] = gt
            rouge = calculate_rouge(self.tokenizer, attacked, batch['input_text'])
        else:
            rouge = calculate_rouge(self.tokenizer, gt, batch['input_text'])
        self.log_to_sample_result(client_id, 'tag_rouge_lf', rouge['rouge-l']['f'])
        self.log_to_all_result(client_id, 'tag_rouge_lf', rouge['rouge-l']['f'])


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)
    # 加载攻击模型
    attacker, attacker2 = get_dra_attacker(get_dra_config(args))

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)

    # 加载TAG攻击模型
    model.config_sfl(config, None)
    tag = get_dlg_attacker(model)

    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    test_dataset = dataset_cls(tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=QAFLStrategy(args, model, tokenizer, test_loader, attacker, attacker2, tag),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)
    # 加载Pre-FT数据集
    if args.pre_ft_dataset is not None and len(args.pre_ft_dataset) > 0:
        pre_ft_dataset = get_dataset_class(args.pre_ft_dataset)(tokenizer=tokenizer, client_ids=[])
        pre_ft_loader = pre_ft_dataset.get_dataloader_unsliced(1, args.pre_ft_data_label,
                                                               shrink_frac=args.pre_ft_data_shrink_frac)
        simulator.pre_ft(pre_ft_loader, ['bottom', 'top'])
    wandb.init(
        project=args.exp_name,
        name=f"C-Dxp{args.noise_scale}-{args.dlg_epochs}-adjust{args.dlg_adjust}-beta{args.dlg_beta}-temp{args.dlg_temp_range}-ft{args.dlg_further_ft}",
        config=args
    )
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    parser.add_argument('--dlg_init_with_dra', type=str2bool, default=False,
                        help='initialize GT vector with DRA attacker')
    parser.add_argument('--dlg_dra_reg', type=float, default=0.0,
                        help='Add regularization term to make GT closer to DRA result')
    parser.add_argument('--dlg_temp_range', type=float, default=0.5)
    parser.add_argument('--dlg_further_ft', type=int, default=0)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
