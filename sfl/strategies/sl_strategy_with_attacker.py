# 定义Client本地学习策略
from typing import Any

import numpy as np
import wandb
from matplotlib import pyplot

from sfl.config import FLConfig
from sfl.strategies.basic import BaseSFLStrategy
from sfl.utils.model import FLConfigHolder, saliency_analysis_direct, Intermediate, evaluate_attacker_rouge, \
    saliency_analysis_generative, draw_generative_saliency_maps, draw_direct_saliency_maps, saliency_analysis_decoder, \
    saliency_analysis_atk, saliency_analysis_atk_mid


class SLStrategyWithAttacker(BaseSFLStrategy):

    def __init__(self, args, fl_config: FLConfig, llm, tokenizer, attackers_conf, **kwargs):
        super().__init__(args, llm, fl_config, tokenizer, **kwargs)
        self.token_class_nums = {}
        self.token_class_hit = {}
        self.attackers = []
        for name, attacker, config_prefix, init in attackers_conf:
            aarg = attacker.parse_arguments(args, config_prefix)
            if aarg.enable:
                self.attackers.append((name, attacker, aarg, init))
                attacker.load_attacker(self.args, aarg, llm, self.tokenizer)

    def client_evaluate(self, global_round, client_id, log):
        super(SLStrategyWithAttacker, self).client_evaluate(global_round, client_id, log)
        self.log_sample_text()

    def log_sample_text(self):
        llm = self.llm
        observe_sample = self.sample_batch
        inverter = self.attackers[0][1].inverter_b2tr
        if inverter is not None:
            inverter.to(llm.device)
            with FLConfigHolder(llm) as ch:
                llm.fl_config.attack_mode = 'b2tr'
                ch.change_config()
                input_tensors = observe_sample['input_ids'].to(llm.device)
                inter = llm(input_tensors)
                attacked = inverter(inter.to(llm.device))
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
                saliency_dir, attacked, stacks_dir = saliency_analysis_atk(llm, inverter, ex_sentence)
                figs = draw_generative_saliency_maps(self.tokenizer, ex_sentence, attacked, stacks_dir)
                wandb.log({'attacked_sample_saliency_matrix_attacked': [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    pyplot.close(fig)
                print('analyzing saliency attacked mid...')
                # attacked_mid
                saliency_dir, attacked, stacks_dir = saliency_analysis_atk_mid(llm, inverter, ex_sentence)
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
        # 逐个调用Attacker
        init_map = {}
        for name, attacker, atk_args, init_from in self.attackers:
            attack_res = attacker.attack(self.args, atk_args, self.llm, self.tokenizer, self.simulator, batch,
                                         b2tr_inter,
                                         tr2t_inter, all_inter, init=init_map.get(init_from, None))
            if isinstance(attack_res, dict):
                for key, logits in attack_res.items():
                    init_map[f"{name}_{key}"] = logits
                    eval_res = evaluate_attacker_rouge(self.tokenizer, logits, batch)
                    self.__log_atk_res(eval_res, client_id, f"{name}_{key}")
            else:
                eval_res = evaluate_attacker_rouge(self.tokenizer, attack_res, batch)
                self.__log_atk_res(eval_res, client_id, name)
                init_map[name] = attack_res

        # 分析Entanglement
        sip_inv_res = init_map.get('SIP_b2tr', None)
        if sip_inv_res is None:
            sip_inv_res = init_map.get('SIP_tr2t', None)
        if self.args.entangle_enable and sip_inv_res is not None:
            saliency, tokens, _ = saliency_analysis_generative(self.llm, batch['input_ids'])
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            saliency_bins = np.digitize(saliency, bins)
            tokens_hit = (sip_inv_res.argmax(-1).cpu() == batch['input_ids']).cpu().numpy()
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
