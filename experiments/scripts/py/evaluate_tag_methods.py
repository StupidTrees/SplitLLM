import os
import sys
from typing import Any

import wandb

sys.path.append(os.path.abspath('../../..'))

from sfl.simulator.strategy import BaseSFLStrategy

from sfl.utils.model import Intermediate, set_random_seed, get_output, evaluate_attacker_rouge
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
            for type, atk in zip(['tr2t', 'b2tr'], [self.dra2, self.dra1]):
                if atk is None:
                    continue
                atk.to(self.simulator.device)
                inter = b2tr_inter if type == 'b2tr' else tr2t_inter
                if self.llm.type == 'encoder-decoder':
                    attacked = atk(torch.concat([encoder_inter.fx.to(
                        self.simulator.device), inter.fx.to(atk.device)], dim=1))
                else:
                    attacked = atk(inter.fx.to(atk.device))
                rouge_res = evaluate_attacker_rouge(self.tokenizer, attacked, batch)
                self.log_to_sample_result(client_id, f'DRA_{type}_rgLf', rouge_res['rouge-l']['f'])
                self.log_to_all_result(client_id, f'DRA_{type}_rgLf', rouge_res['rouge-l']['f'])
                logs[f'attacker_{type}_step'] = rouge_res['rouge-l']['f']
        if self.args.dlg_enable:
            gt_init = None
            if self.args.dlg_init_with_dra:
                gt_init = attacked
            self.dlg.to(self.simulator.device)
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
            dlg_logits = gt
            if self.llm.type == 'encoder-decoder':
                dlg_logits = attacked.clone()
                dlg_logits[:, -gt.shape[1]:, :] = gt
            dlg_rouge = evaluate_attacker_rouge(self.tokenizer, dlg_logits, batch)
            self.log_to_sample_result(client_id, 'DLG_rgL_f', dlg_rouge['rouge-l']['f'])
            self.log_to_all_result(client_id, 'DLG_rgL_f', dlg_rouge['rouge-l']['f'])
            if args.dlg_raw_enable:  # 进行不初始化的dlg以作为对比
                gt2 = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device), tr2t_inter.grad.to(self.simulator.device),
                                   epochs=self.args.dlg_epochs * 8,  # 轮次要拉大一点
                                   beta=self.args.dlg_beta,
                                   gt_init=None,
                                   encoder_inter=None if encoder_inter is None else encoder_inter.fx.to(
                                       self.simulator.device)
                                   )
                dlg2_logits = gt2
                if self.llm.type == 'encoder-decoder':
                    dlg2_logits = attacked.clone()
                    dlg2_logits[:, -gt2.shape[1]:, :] = gt2
                dlg2_rouge = evaluate_attacker_rouge(self.tokenizer, dlg2_logits, batch)
                self.log_to_sample_result(client_id, 'DLG_raw_rgLf', dlg2_rouge['rouge-l']['f'])
                self.log_to_all_result(client_id, 'DLG_raw_rgLf', dlg2_rouge['rouge-l']['f'])


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)

    # 加载TAG攻击模型
    model.config_sfl(config, None)
    model.train()
    print(get_output("test", tokenizer, model))
    tag = get_dlg_attacker(model)

    # 加载DRA攻击模型
    attacker, attacker2 = get_dra_attacker(get_dra_config(args))

    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, args.test_data_label,
                                                       shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=QAFLStrategy(args, model, tokenizer, test_loader, attacker, attacker2, tag),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)
    # 加载Pre-FT数据集
    if args.pre_ft_dataset is not None and len(args.pre_ft_dataset) > 0:
        pre_ft_dataset = get_dataset(args.pre_ft_dataset, tokenizer=tokenizer, client_ids=[])
        pre_ft_loader = pre_ft_dataset.get_dataloader_unsliced(args.batch_size, args.pre_ft_data_label,
                                                               shrink_frac=args.pre_ft_data_shrink_frac)
        simulator.pre_ft(pre_ft_loader, ['bottom', 'top'])

    noise_name = args.noise_mode
    if args.noise_mode == 'grad':
        noise_name += f'[{args.noise_scale_grad}]'
    elif args.noise_mode == 'dxp':
        noise_name += f'[{args.noise_scale_dxp}]'
    elif args.noise_mode == 'both':
        noise_name += f'[{args.noise_scale_dxp},{args.noise_scale_grad}]'

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
