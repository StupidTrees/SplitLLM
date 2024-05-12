
# 定义Client本地学习策略
from typing import Any

import torch
import wandb
from matplotlib import pyplot

from sfl.model.attacker.eia_attacker import WhiteBoxMIAttacker
from sfl.simulator.simulator import ParamRestored
from sfl.strategies.basic import BaseSFLStrategy
from sfl.utils.exp import get_eia_mapper, get_mapper_config
from sfl.utils.model import FLConfigHolder, saliency_analysis_direct, Intermediate, evaluate_attacker_rouge, \
    saliency_analysis_generative, draw_generative_saliency_maps, draw_direct_saliency_maps, saliency_analysis_decoder, \
    saliency_analysis_atk, saliency_analysis_atk_mid


class SLStrategyWithAttacker(BaseSFLStrategy):

    def __init__(self, args, llm, tokenizer, **kwargs):
        super().__init__(args, llm, tokenizer, **kwargs)
        self.token_class_nums = {}
        self.token_class_hit = {}

    def client_evaluate(self, global_round, client_id, log):
        super(SLStrategyWithAttacker, self).client_evaluate(global_round, client_id, log)
        self.log_sample_text()

    def log_sample_text(self):
        llm = self.llm
        observe_sample = self.sample_batch
        if self.dra1 is not None:
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
            ex_sentence = observe_sample['input_ids'][:, :32]
            if self.args.entangle_enable:
                print('analyzing saliency Generative...')
                saliency, next_token_ids, stacks = saliency_analysis_generative(llm, input_sentence)
                # saliency = saliency.detach().cpu().numpy()

                # generative
                figs = draw_generative_saliency_maps(self.tokenizer,
                                                     input_sentence,
                                                     next_token_ids,
                                                     stacks)
                wandb.log({'attacked_sample_saliency_matrix_generative': [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    pyplot.close(fig)
                table = wandb.Table(
                    columns=["attacked_text", "true_text", "input_text", "generated_text", "input_tokens",
                             "generated_tokens",
                             "avg_saliency"],
                    data=[[txt, gt, self.tokenizer.decode(it, skip_special_tokens=True),
                           self.tokenizer.decode(nt, skip_special_tokens=True),
                           self.tokenizer.convert_ids_to_tokens(it),
                           self.tokenizer.convert_ids_to_tokens(nt),
                           sa.tolist()]
                          for txt, gt, it, nt, sa, stack in
                          zip(texts, observe_sample['input_text'], input_sentence, next_token_ids, saliency,
                              stacks)])
                print('analyzing saliency Direct...')
                # direct
                saliency_dir, stacks_dir = saliency_analysis_direct(llm, ex_sentence)
                figs = draw_direct_saliency_maps(self.tokenizer, ex_sentence, stacks_dir)
                wandb.log({'attacked_sample_saliency_matrix_direct': [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    pyplot.close(fig)
                print('analyzing saliency Decoder...')
                # decoder
                saliency_dir, stacks_dir = saliency_analysis_decoder(llm, ex_sentence)
                figs = draw_direct_saliency_maps(self.tokenizer, ex_sentence, stacks_dir)
                wandb.log({'attacked_sample_saliency_matrix_decoder': [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    pyplot.close(fig)
                print('analyzing saliency attacked...')
                # attacked
                saliency_dir, attacked, stacks_dir = saliency_analysis_atk(llm, self.dra1, ex_sentence)
                figs = draw_generative_saliency_maps(self.tokenizer, ex_sentence, attacked, stacks_dir)
                wandb.log({'attacked_sample_saliency_matrix_attacked': [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    pyplot.close(fig)
                print('analyzing saliency attacked mid...')
                # attacked_mid
                saliency_dir, attacked, stacks_dir = saliency_analysis_atk_mid(llm, self.dra1, ex_sentence)
                figs = draw_generative_saliency_maps(self.tokenizer, ex_sentence, attacked, stacks_dir)
                wandb.log({'attacked_sample_saliency_matrix_attacked_mid': [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    pyplot.close(fig)
            else:
                table = wandb.Table(
                    columns=["attacked_text", "true_text", "input_text"],
                    data=[[txt, gt, self.tokenizer.decode(it, skip_special_tokens=True),
                           ]
                          for txt, gt, it in
                          zip(texts, observe_sample['input_text'], input_sentence)])

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
                if type == 'b2tr':
                    if self.args.attacker_b2tr_target_sp >= 0:
                        inter = all_inter[self.args.attacker_b2tr_target_sp]
                    else:
                        inter = b2tr_inter
                else:
                    if self.args.attacker_tr2t_target_sp >= 0:
                        inter = all_inter[self.args.attacker_tr2t_target_sp]
                    else:
                        inter = tr2t_inter
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
            atk_init = attacked_result.get('b2tr', None)
        elif self.args.attacker_tr2t_sp == self.simulator.config.split_point_2:
            atk_init = attacked_result.get('tr2t', None)
        else:
            atk_init = attacked_result.get('b2tr', None)

        # 分析Entanglement
        if self.args.entangle_enable:
            saliency, tokens, _ = saliency_analysis_generative(self.llm, batch['input_ids'])
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

        atk_init_before_dlg = atk_init
        if self.args.alt_enable:
            # 交替优化
            alt_out = atk_init
            for i in range(self.args.alt_steps):
                atk = WhiteBoxMIAttacker()
                with ParamRestored(llm=self.llm, param_keeper=self.simulator.parameter_keeper,
                                   key='pretrained',
                                   parts=['bottom']):
                    alt_out = atk.fit(tok=self.tokenizer,
                                      llm=self.llm, inter=b2tr_inter, gt=batch,
                                      dummy_init=alt_out, epochs=self.args.alt_fwd_steps)
                alt_out = alt_out.argmax(-1)
                # one-hot
                alt_out = torch.nn.functional.one_hot(alt_out, num_classes=self.llm.config.vocab_size).float()
                alt_out = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device),
                                       tr2t_inter.grad.to(self.simulator.device),
                                       epochs=self.args.alt_bwd_steps,
                                       beta=self.args.dlg_beta,
                                       gt_init=alt_out,
                                       lr=self.args.dlg_lr,
                                       init_temp=self.args.dlg_init_temp,
                                       softmax=True,
                                       encoder_inter=None if encoder_inter is None else encoder_inter.fx.to(
                                           self.simulator.device),
                                       model_name=self.args.model_name,
                                       attention_mask=attention_mask.fx if attention_mask is not None else None,
                                       rotary_pos_emb=rotary_pos_emb.fx if rotary_pos_emb is not None else None,
                                       )

            eval_logits = alt_out
            if self.llm.type == 'encoder-decoder':
                eval_logits = atk_init.clone()
                eval_logits[:, -alt_out.shape[1]:, :] = alt_out
            alt_res = evaluate_attacker_rouge(self.tokenizer, eval_logits, batch)
            self.__log_atk_res(alt_res, client_id, 'BiSR+')
        else:
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
                                                     lr=self.args.dlg_lr,
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
                                                     lr=self.args.dlg_lr,
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
                    gt = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device),
                                      tr2t_inter.grad.to(self.simulator.device),
                                      epochs=self.args.dlg_epochs,
                                      adjust=self.args.dlg_adjust,
                                      beta=self.args.dlg_beta,
                                      gt_init=gt_init,
                                      lr=self.args.dlg_lr,
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

                if self.args.dlg_raw_enable:  # 进行不初始化的dlg以作为对比
                    gt2 = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device),
                                       tr2t_inter.grad.to(self.simulator.device),
                                       epochs=self.args.dlg_raw_epochs,  # self.args.dlg_epochs ,  # 轮次要拉大一点
                                       beta=self.args.dlg_beta,
                                       gt_init=None,
                                       lr=self.args.dlg_lr,
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
                parts = ['bottom']
                inter = b2tr_inter
                if self.args.wba_at == 'tr2t':
                    inter = tr2t_inter
                with ParamRestored(llm=self.llm, param_keeper=self.simulator.parameter_keeper, key='pretrained',
                                   parts=parts):
                    spis_out = atk.fit(tok=self.tokenizer,
                                       llm=self.llm, inter=inter, gt=batch,
                                       dummy_init=atk_init, epochs=self.args.wba_epochs,
                                       lr=self.args.wba_lr, cos_loss=True, at=self.args.wba_at)
                wba_res = evaluate_attacker_rouge(self.tokenizer, spis_out, batch)
                self.__log_atk_res(wba_res, client_id, f'BiSR+')
            if self.args.wba_dir_enable:
                atk = WhiteBoxMIAttacker()
                parts = ['bottom']
                inter = b2tr_inter
                if self.args.wba_at == 'tr2t':
                    inter = tr2t_inter
                with ParamRestored(llm=self.llm, param_keeper=self.simulator.parameter_keeper, key='pretrained',
                                   parts=parts):
                    spis_out = atk.fit(tok=self.tokenizer,
                                       llm=self.llm, inter=inter, gt=batch,
                                       dummy_init=atk_init_before_dlg, epochs=self.args.wba_epochs,
                                       lr=self.args.wba_lr, cos_loss=True, at=self.args.wba_at)
                wba_res = evaluate_attacker_rouge(self.tokenizer, spis_out, batch)
                self.__log_atk_res(wba_res, client_id, f'BiSR-')
            if self.args.wba_raw_enable:
                atk = WhiteBoxMIAttacker()
                mapper = None
                parts = ['bottom']
                inter = b2tr_inter
                if self.args.wba_at == 'tr2t':
                    inter = tr2t_inter
                if self.args.wba_raw_mapped_to > 0:
                    mapper = get_eia_mapper(get_mapper_config(self.args)).to(self.llm.device)
                    # inter = all_inter.get(self.args.wba_raw_mapped_to - 1, None)
                with ParamRestored(llm=self.llm, param_keeper=self.simulator.parameter_keeper, key='pretrained',
                                   parts=parts):
                    wba_raw_out = atk.fit_eia(tok=self.tokenizer, mapper=mapper,mapped_to=self.args.wba_raw_mapped_to,
                                              llm=self.llm, inter=inter, gt=batch, lr=self.args.wba_raw_lr,
                                              dummy_init=None, epochs=self.args.wba_raw_epochs, at=self.args.wba_at,
                                              temp=self.args.wba_raw_temp, wd=self.args.wba_raw_wd)
                wba_raw_res = evaluate_attacker_rouge(self.tokenizer, wba_raw_out, batch)
                self.__log_atk_res(wba_raw_res, client_id, f'MIA')
            # wba_out = self.simulator.restored_run(atk.fit, key='pretrained', parts=['bottom', 'top'],
            #                                       tok=self.tokenizer,
            #                                       llm=self.llm, b2tr_inter=b2tr_inter, gt=batch)
            # wba_res = evaluate_attacker_rouge(self.tokenizer, wba_out, batch)
            # self.__log_atk_res(wba_res, client_id, f'WBA')
