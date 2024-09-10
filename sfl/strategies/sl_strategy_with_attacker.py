from typing import Any

import wandb

from sfl.utils.args import FLConfig
from sfl.strategies.basic import BaseSFLStrategy
from sfl.utils.model import FLConfigHolder, Intermediate, evaluate_attacker_rouge


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
        self.log_all_sample_text()

    def log_all_sample_text(self):
        llm = self.llm
        batch = self.sample_batch
        training = llm.training
        with FLConfigHolder(llm) as ch:
            llm.training = True
            llm.fl_config.collect_intermediates = True
            llm.fl_config.collect_all_intermediates = True
            ch.change_config()
            input_ids = batch['input_ids'].to(llm.device)
            outputs = llm(input_ids=input_ids, attention_mask=batch['attention_mask'].to(llm.device), labels=input_ids)
            outputs.loss.backward()
            b2tr_inter, tr2t_inter, all_inters = llm.get_all_inter()
            init_map = {}
            all_tables = {}
            for name, attacker, atk_args, init_from in self.attackers:
                attack_res = attacker.attack(self.args, atk_args, self.llm, self.tokenizer, self.simulator, batch,
                                             b2tr_inter,
                                             tr2t_inter, all_inters, init=init_map.get(init_from, None))
                if isinstance(attack_res, dict):
                    for key, logits in attack_res.items():
                        init_map[f"{name}_{key}"] = logits
                        all_tables[f"{name}_{key}"] = self.__get_atk_example_table(logits, batch)
                else:
                    init_map[name] = attack_res
                    all_tables[name] = self.__get_atk_example_table(attack_res, batch)

            wandb.log({f'sample_text_{key}': table for key, table in all_tables.items()})
            llm.train(training)

    def __log_atk_res(self, attacker_res, client_id, name):
        rouge_res, meteor_res, token_acc = attacker_res
        for key in rouge_res.keys():
            self.log_to_sample_result(client_id, f'{name}_{key}_f', rouge_res[key]['f'])
            self.log_to_all_result(client_id, f'{name}_{key}_f', rouge_res[key]['f'])
        self.log_to_sample_result(client_id, f'{name}_METEOR', meteor_res)
        self.log_to_all_result(client_id, f'{name}_METEOR', meteor_res)
        self.log_to_sample_result(client_id, f'{name}_TOKACC', token_acc)
        self.log_to_all_result(client_id, f'{name}_TOKACC', token_acc)

    def __get_atk_example_table(self, attacked_logits, batch):
        texts = [self.tokenizer.decode(t.argmax(-1), skip_special_tokens=True) for t in attacked_logits]
        table = wandb.Table(columns=["attacked_text", "true_text"],
                            data=[[txt, gt] for txt, gt in zip(texts, batch['input_text'])])
        return table

    def sample_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[Any, Intermediate],
                                  batch, logs):
        # Call attackers sequentially
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