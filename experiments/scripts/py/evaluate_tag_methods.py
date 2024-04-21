import os
import sys
from typing import Any

import numpy as np
import wandb
from matplotlib import pyplot

sys.path.append(os.path.abspath('../../..'))

from sfl.simulator.strategy import BaseSFLStrategy
from sfl.model.attacker.mia_attacker import WhiteBoxMIAttacker

from sfl.utils.model import Intermediate, set_random_seed, get_output, evaluate_attacker_rouge, FLConfigHolder, \
    saliency_analysis
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *


# 定义Client本地学习策略
class QAFLStrategy(BaseSFLStrategy):

    def __init__(self, args, llm, tokenizer, **kwargs):
        super().__init__(args, llm, tokenizer, **kwargs)
        self.token_class_nums = {}
        self.token_class_hit = {}

    def client_evaluate(self, global_round, client_id, log):
        # super(QAFLStrategy, self).client_evaluate(global_round, client_id, log)
        self.log_sample_text()

    def log_sample_text(self):
        llm = self.llm
        observe_sample = self.sample_batch
        self.dra1.to(llm.device)
        with FLConfigHolder(llm) as ch:
            llm.fl_config.attack_mode = 'b2tr'
            ch.change_config()
            input_tensors = observe_sample['input_ids'].to(llm.device)
            inter = llm(input_tensors)
            attacked = self.dra1(inter.to(llm.device))
            # ids = [','.join([str(tid.item()) for tid in t.argmax(-1)]) for t in attacked]
            # true_ids = [','.join([str(tid.item()) for tid in t]) for t in observe_sample['input_ids']]
            texts = [self.tokenizer.decode(t.argmax(-1), skip_special_tokens=True) for t in attacked]
        input_sentence = observe_sample['input_ids']
        if 'q_ids' in observe_sample:
            input_sentence = observe_sample['q_ids']

        saliency, next_token_ids, stacks = saliency_analysis(llm, input_sentence)
        # saliency = saliency.detach().cpu().numpy()
        images = []
        for i, (q, a, saliency_matrix) in enumerate(zip(input_sentence, next_token_ids, stacks)):
            # plot heatmap on saliency_matrix and log to wandb
            fig, ax = pyplot.subplots()
            fig.set_size_inches(16, 16)
            cax = ax.matshow(saliency_matrix, cmap='hot', vmin=0, vmax=1)
            # scale it to square
            ax.set_aspect('auto')
            fig.colorbar(cax)
            print(len(self.tokenizer.convert_ids_to_tokens(q)),
                  len(self.tokenizer.convert_ids_to_tokens(a)),
                  saliency_matrix.shape)

            ax.set_xticks(ticks=range(len(q)), labels=self.tokenizer.convert_ids_to_tokens(q))
            ax.set_yticks(ticks=range(len(a)), labels=self.tokenizer.convert_ids_to_tokens(a))

            # ax.set_yticklabels(self.tokenizer.convert_ids_to_tokens(a), rotation=45)
            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            ax.set_title('Saliency Matrix')
            images.append(wandb.Image(fig))
        wandb.log({'attacked_sample_saliency_matrix': images})

        table = wandb.Table(
            columns = ["attacked_text", "true_text", "input_text", "generated_text", "input_tokens", "generated_tokens",
                       "avg_saliency", "saliency_heatmap"],
            data = [[txt, gt, self.tokenizer.decode(it, skip_special_tokens=True),
                     self.tokenizer.decode(nt, skip_special_tokens=True),
                     self.tokenizer.convert_ids_to_tokens(it),
                     self.tokenizer.convert_ids_to_tokens(nt),
                     sa.tolist(), img]
                    for txt, gt, it, nt, sa, img, stack in
                    zip(texts, observe_sample['input_text'], input_sentence, next_token_ids, saliency, images, stacks)])

        wandb.log({'attacked_sample_text': table})

    def __log_atk_res(self, attacker_res, client_id, name):
        rouge_res, meteor_res, token_acc = attacker_res
        for key in rouge_res.keys():
            self.log_to_sample_result(client_id, f'{name}_{key}_f', rouge_res[key]['f'])
            self.log_to_all_result(client_id, f'{name}_{key}_f', rouge_res[key]['f'])
        self.log_to_sample_result(client_id, f'{name}_METEOR', meteor_res)
        self.log_to_all_result(client_id, f'{name}_METEOR', meteor_res)
        self.log_to_sample_result(client_id, f'{name}_TOKACC', token_acc)
        self.log_to_all_result(client_id, f'{name}_TOKACC', token_acc)

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
                inter = b2tr_inter if type == 'b2tr' else tr2t_inter
                if self.llm.type == 'encoder-decoder':
                    attacked = atk(torch.concat([encoder_inter.fx.to(
                        self.simulator.device), inter.fx.to(atk.device)], dim=1))
                else:
                    attacked = atk(inter.fx.to(atk.device))
                sip_res = evaluate_attacker_rouge(self.tokenizer, attacked, batch)
                self.__log_atk_res(sip_res, client_id, f'SIP_{type}')
                # logs[f'SIP_{type}_rouge-l_step'] = rouge_res['rouge-l']['f']
                attacked_result[type] = attacked

        if self.args.attacker_b2tr_sp == self.simulator.config.split_point_1:
            atk_init = attacked_result['b2tr']
        elif self.args.attacker_tr2t_sp == self.simulator.config.split_point_2:
            atk_init = attacked_result['tr2t']
        else:
            atk_init = attacked_result['b2tr']

        # 分析Entanglement
        if self.args.entangle_enable:
            saliency, tokens, _ = saliency_analysis(self.llm, batch['input_ids'])
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            saliency_bins = np.digitize(saliency, bins)
            tokens_hit = (atk_init.argmax(-1).cpu() == batch['input_ids']).cpu().numpy()
            for i, (s, h) in enumerate(zip(saliency_bins, tokens_hit)):
                for cid, hit in zip(s, h):
                    if cid not in self.token_class_nums:
                        self.token_class_nums[cid] = 0
                        self.token_class_hit[cid] = 0
                    self.token_class_nums[cid] += 1
                    self.token_class_hit[cid] += hit
            for cid, hit in self.token_class_hit.items():
                self.log_to_sample_result(client_id, f'hit_rate_{cid}', hit / self.token_class_nums[cid])
                self.log_to_all_result(client_id, f'hit_rate_{cid}', hit / self.token_class_nums[cid])

        if self.args.dlg_enable:
            gt_init = None
            if self.args.dlg_init_with_dra:
                gt_init = atk_init
            self.dlg.to(self.simulator.device)
            gma_name = 'BiSR'
            if self.dlg.method == 'lamp':
                gt = self.simulator.restored_run(self.dlg.fit, key='pretrained', parts=['bottom', 'top', 'trunk'],
                                                 write_back=False,
                                                 epochs=self.args.dlg_epochs,
                                                 lr=args.dlg_lr,
                                                 inter=tr2t_inter.fx.clone().to(self.simulator.device),
                                                 gradient=tr2t_inter.grad.clone().to(self.simulator.device),
                                                 beta=self.args.dlg_beta,
                                                 gt_init=None,
                                                 model_name=self.args.model_name,
                                                 encoder_inter=None if encoder_inter is None else encoder_inter.fx.to(
                                                     self.simulator.device),
                                                 attention_mask=attention_mask.fx if attention_mask is not None else None,
                                                 rotary_pos_emb=rotary_pos_emb.fx if rotary_pos_emb is not None else None,
                                                 lamp=True,
                                                 lamp_freq=self.args.dlg_lamp_freq
                                                 )
                # lamp_res = evaluate_attacker_rouge(self.tokenizer, gt, batch)
                # self.__log_atk_res(lamp_res, client_id, 'LAMP')
                gma_name = 'LAMP'
            elif self.dlg.method == 'bisr':
                gt = self.simulator.restored_run(self.dlg.fit, key='pretrained', parts=['bottom', 'top'],
                                                 write_back=False,
                                                 epochs=self.args.dlg_epochs,
                                                 lr=args.dlg_lr,
                                                 inter=tr2t_inter.fx.clone().to(self.simulator.device),
                                                 gradient=tr2t_inter.grad.clone().to(self.simulator.device),
                                                 beta=self.args.dlg_beta,
                                                 gt_init=gt_init,
                                                 model_name=self.args.model_name,
                                                 encoder_inter=None if encoder_inter is None else encoder_inter.fx.to(
                                                     self.simulator.device),
                                                 attention_mask=attention_mask.fx if attention_mask is not None else None,
                                                 rotary_pos_emb=rotary_pos_emb.fx if rotary_pos_emb is not None else None,
                                                 lamp=True,
                                                 lamp_freq=self.args.dlg_lamp_freq
                                                 )
                # lamp_res = evaluate_attacker_rouge(self.tokenizer, gt, batch)
                # self.__log_atk_res(lamp_res, client_id, 'LAMP')
                gma_name = 'BiSR'
            else:
                gt = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device), tr2t_inter.grad.to(self.simulator.device),
                                  epochs=self.args.dlg_epochs,
                                  adjust=args.dlg_adjust,
                                  beta=self.args.dlg_beta,
                                  gt_init=gt_init,
                                  lr=args.dlg_lr,
                                  init_temp=self.args.dlg_init_temp,
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
            atk_init = dlg_logits
            dlg_res = evaluate_attacker_rouge(self.tokenizer, dlg_logits, batch)
            self.__log_atk_res(dlg_res, client_id, gma_name)

            if args.dlg_raw_enable:  # 进行不初始化的dlg以作为对比
                gt2 = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device), tr2t_inter.grad.to(self.simulator.device),
                                   epochs=self.args.dlg_raw_epochs,  # self.args.dlg_epochs ,  # 轮次要拉大一点
                                   beta=self.args.dlg_beta,
                                   gt_init=None,
                                   lr=args.dlg_lr,
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
                tag_res = evaluate_attacker_rouge(self.tokenizer, dlg2_logits, batch)
                self.__log_atk_res(tag_res, client_id, 'TAG')

        if self.args.wba_enable:
            atk = WhiteBoxMIAttacker()
            spis_out = self.simulator.restored_run(atk.fit, key='pretrained', parts=['bottom', 'top'],
                                                   tok=self.tokenizer,
                                                   llm=self.llm, b2tr_inter=b2tr_inter, gt=batch,
                                                   dummy_init=atk_init, epochs=200)
            wba_res = evaluate_attacker_rouge(self.tokenizer, spis_out, batch)
            self.__log_atk_res(wba_res, client_id, f'SIPS')
            # wba_out = self.simulator.restored_run(atk.fit, key='pretrained', parts=['bottom', 'top'],
            #                                       tok=self.tokenizer,
            #                                       llm=self.llm, b2tr_inter=b2tr_inter, gt=batch)
            # wba_res = evaluate_attacker_rouge(self.tokenizer, wba_out, batch)
            # self.__log_atk_res(wba_res, client_id, f'WBA')


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
                                                       shrink_frac=args.test_data_shrink_frac,
                                                       max_seq_len=args.dataset_max_seq_len)
    sample_batch = next(iter(fed_dataset.get_dataloader_unsliced(3, 'train')))
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=QAFLStrategy(args, model, tokenizer,
                                                   test_loader=test_loader,
                                                   sample_batch=sample_batch,
                                                   dra1=attacker,
                                                   dra2=attacker2,
                                                   dlg=tag),
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
