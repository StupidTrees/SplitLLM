import os
import sys
from copy import deepcopy
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
        attention_mask = all_inter.get('att_msk', None)
        rotary_pos_emb = all_inter.get('rot_pos', None)
        attacked_result = {}
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
                self.log_to_sample_result(client_id, f'DRA_{type}_rg1f', rouge_res['rouge-1']['f'])
                self.log_to_all_result(client_id, f'DRA_{type}_rg1f', rouge_res['rouge-1']['f'])
                logs[f'attacker_{type}_step'] = rouge_res['rouge-l']['f']
                attacked_result[type] = attacked
        if self.args.dlg_enable:
            gt_init = None
            if self.args.dlg_init_with_dra:
                if self.args.attacker_b2tr_sp == self.simulator.config.split_point_1:
                    gt_init = attacked_result['b2tr']
                elif self.args.attacker_tr2t_sp == self.simulator.config.split_point_2:
                    gt_init = attacked_result['tr2t']
                else:
                    gt_init = attacked_result['b2tr']
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
                                  self.simulator.device),
                              model_name=self.args.model_name,
                              attention_mask=attention_mask.fx if attention_mask is not None else None,
                              rotary_pos_emb=rotary_pos_emb.fx if rotary_pos_emb is not None else None,
                              )
            dlg_logits = gt
            if self.llm.type == 'encoder-decoder':
                dlg_logits = attacked.clone()
                dlg_logits[:, -gt.shape[1]:, :] = gt
            dlg_rouge = evaluate_attacker_rouge(self.tokenizer, dlg_logits, batch)
            self.log_to_sample_result(client_id, 'DLG_rgL_f', dlg_rouge['rouge-l']['f'])
            self.log_to_sample_result(client_id, 'DLG_rg1_f', dlg_rouge['rouge-1']['f'])
            self.log_to_all_result(client_id, 'DLG_rg1_f', dlg_rouge['rouge-1']['f'])
            self.log_to_all_result(client_id, 'DLG_rgL_f', dlg_rouge['rouge-l']['f'])
            if self.dlg.method == 'lamp':
                # clear torch cache to avoid memory leak
                torch.cuda.empty_cache()
                gt = self.simulator.restored_run(self.dlg.lamp, key='pretrained', write_back=False,
                                                 inter=tr2t_inter.fx.to(self.simulator.device),
                                                 gradient=tr2t_inter.grad.to(self.simulator.device),
                                                 gt=deepcopy(gt.detach()),
                                                 beta=self.args.dlg_beta)
                dlg_rouge = evaluate_attacker_rouge(self.tokenizer, gt, batch)
                self.log_to_sample_result(client_id, 'LAMP_rgL_f', dlg_rouge['rouge-l']['f'])
                self.log_to_sample_result(client_id, 'LAMP_rg1_f', dlg_rouge['rouge-1']['f'])
                self.log_to_all_result(client_id, 'LAMP_rgL_f', dlg_rouge['rouge-l']['f'])
                self.log_to_all_result(client_id, 'LAMP_rg1_f', dlg_rouge['rouge-1']['f'])

            if args.dlg_raw_enable:  # 进行不初始化的dlg以作为对比
                gt2 = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device), tr2t_inter.grad.to(self.simulator.device),
                                   epochs=200,  # self.args.dlg_epochs ,  # 轮次要拉大一点
                                   beta=self.args.dlg_beta,
                                   gt_init=None,
                                   model_name=self.args.model_name,
                                   encoder_inter=None if encoder_inter is None else encoder_inter.fx.to(
                                       self.simulator.device),
                                   attention_mask=attention_mask.fx if attention_mask is not None else None,
                                   rotary_pos_emb=rotary_pos_emb.fx if rotary_pos_emb is not None else None,
                                   )
                dlg2_logits = gt2
                if self.llm.type == 'encoder-decoder':
                    dlg2_logits = attacked.clone()
                    dlg2_logits[:, -gt2.shape[1]:, :] = gt2
                dlg2_rouge = evaluate_attacker_rouge(self.tokenizer, dlg2_logits, batch)
                self.log_to_sample_result(client_id, 'DLG_raw_rgLf', dlg2_rouge['rouge-l']['f'])
                self.log_to_sample_result(client_id, 'DLG_raw_rg1f', dlg2_rouge['rouge-1']['f'])
                self.log_to_all_result(client_id, 'DLG_raw_rgLf', dlg2_rouge['rouge-l']['f'])
                self.log_to_all_result(client_id, 'DLG_raw_rg1f', dlg2_rouge['rouge-1']['f'])


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)

    # 加载TAG攻击模型
    model.config_sfl(config, None)
    model.train()
    tag = None
    if args.dlg_enable:
        print(get_output("test", tokenizer, model))
        tag = get_dlg_attacker(model, method=args.dlg_method)

    # 加载DRA攻击模型
    attacker, attacker2 = get_dra_attacker(get_dra_config(args))
    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, args.test_data_label,
                                                       shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=QAFLStrategy(args, model, tokenizer, test_loader, attacker, attacker2, tag),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)
    if tag is not None:
        tag.simulator = simulator
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
