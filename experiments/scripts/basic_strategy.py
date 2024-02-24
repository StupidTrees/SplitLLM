from typing import Iterator

import torch
from numpy import average
from tqdm import tqdm
from transformers import AdamW

from sfl.config import FLConfig, Intermediate
from sfl.model.split_model import SplitModel
from sfl.simulator.strategy import FLStrategy
from sfl.utils.model import calculate_rouge
from sfl.utils.training import evaluate_perplexity, evaluate_accuracy


class BaseSFLStrategy(FLStrategy):

    def __init__(self, args, tokenizer, attacker1, attacker2, llm, test_loader):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.attacker1 = attacker1
        self.attacker2 = attacker2
        self.test_loader = test_loader
        self.dlg = None
        self.llm = llm
        self.attack_sample_counter = {}
        self.attack_sample_performs = {}
        self.attack_all_performs = {}

    def client_evaluate(self, global_round, client_id, log):
        if self.task_type == 'clsf':
            log['test-acc'] = evaluate_accuracy(self.simulator.llm, self.test_loader)
        elif self.task_type == 'lm':
            ppl = evaluate_perplexity(self.simulator.llm, self.test_loader)
            log['test-ppl'] = ppl

    def client_step(self, client_id: str, global_round, client_epoch, llm: SplitModel, iterator: Iterator,
                    config: FLConfig):
        optimizer = AdamW(llm.parameters(), lr=1e-5)
        avg_loss = 0
        # avg_self_rouge = 0
        # avg_self_rouge_pt = 0
        batch_num = 0
        with tqdm(total=config.client_steps) as pbar:
            for step, batch in enumerate(iterator):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(llm.device)
                attention_mask = batch['input_att_mask'].to(llm.device)
                labels = input_ids
                if 'labels' in batch and self.task_type == 'clsf':
                    labels = batch['labels'].to(llm.device)
                outputs = llm(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                pbar.set_description(
                    f'Client {client_id} Epoch {client_epoch} Step {self.simulator.get_current_step(client_id, step)} Loss {loss.item():.3f}')
                loss.backward()
                batch_num += 1
                optimizer.step()
                avg_loss += loss.detach().cpu().item()
                # avg_self_rouge += \
                #     calculate_rouge(self.tokenizer, outputs.logits.argmax(dim=-1), batch['input_text'])['rouge-l'][
                #         'f']
                # if self.args.self_pt_enable:
                #     outputs_pt = self.simulator.restored_forward('top', input_ids=input_ids, labels=input_ids,
                #                                                  attention_mask=attention_mask)
                #     avg_self_rouge_pt += \
                #         calculate_rouge(self.tokenizer, outputs_pt.logits, batch['input_text'])['rouge-l']['f']
                self.step_done(client_id, step, batch,
                               {"loss": float(avg_loss / batch_num)})
                # , "self": float(avg_self_rouge / batch_num),
                #  "self_pt": float(avg_self_rouge_pt / batch_num)})  # Collect gradients
                pbar.update(1)

    def callback_intermediate_result(self, global_round, client_id, local_epoch, local_step,
                                     b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                     all_inter: dict[int, Intermediate],
                                     batch, logs
                                     ):
        # 一定频率进行攻击
        self.attack_all_performs.setdefault(client_id, {})
        # 每batch触发攻击
        self.normal_attacker_triggered(global_round, client_id, local_epoch, local_step, b2tr_inter, tr2t_inter,
                                       all_inter, batch, logs)
        if (local_step + 1) % self.args.attacker_freq == 0:
            self.attack_sample_counter[client_id] = 0
            self.attack_sample_performs[client_id] = {}

        if client_id in self.attack_sample_counter:
            self.attack_sample_counter[client_id] += 1
            self.sample_attacker_triggered(global_round, client_id, local_epoch, local_step, b2tr_inter, tr2t_inter,
                                           all_inter,
                                           batch, logs)
            for key, list in self.attack_all_performs[client_id].items():
                logs[key + '_avg'] = average(list)
            if self.attack_sample_counter[client_id] >= self.args.attacker_samples:
                del self.attack_sample_counter[client_id]
                for key, list in self.attack_sample_performs[client_id].items():
                    logs[key + '_sampled'] = average(list)
                del self.attack_sample_performs[client_id]

    def log_to_sample_result(self, client_id, key, value):
        self.attack_sample_performs[client_id].setdefault(key, [])
        self.attack_sample_performs[client_id][key].append(value)

    def log_to_all_result(self, client_id, key, value):
        self.attack_all_performs[client_id].setdefault(key, [])
        self.attack_all_performs[client_id][key].append(value)

    def normal_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[int, Intermediate],
                                  batch, logs):
        """
        每个batch，都触发这个攻击函数
        """
        pass

    def sample_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[int, Intermediate],
                                  batch, logs):
        """
        一定频率，触发这个攻击函数
        """
        if self.dlg is not None:
            self.dlg.to(self.simulator.device)
            gt = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device), tr2t_inter.grad, epochs=self.args.dlg_epochs,
                              adjust=self.args.dlg_adjust, beta=self.args.dlg_beta)
            rouge = calculate_rouge(self.tokenizer, gt, batch['input_text'])
            self.log_to_sample_result(client_id, 'tag_rouge_lf', rouge['rouge-l']['f'])
            self.log_to_all_result(client_id, 'tag_rouge_lf', rouge['rouge-l']['f'])
        rg = [False]
        if self.args.attacker_search:
            rg.append(True)
        for search in rg:
            suffix = f'{"search" if search else "normal"}'
            with torch.no_grad():
                for type, atk in zip(['b2tr', 'tr2t'], [self.attacker1, self.attacker2]):
                    if atk is None:
                        continue
                    atk.to(self.simulator.device)
                    inter = b2tr_inter if type == 'b2tr' else tr2t_inter
                    if search:
                        attacked = atk.search(inter.fx, self.llm)
                    else:
                        attacked = atk(inter.fx.to(atk.device))
                    rouge_res = calculate_rouge(self.tokenizer, attacked, batch['input_text'])
                    self.log_to_sample_result(client_id, f'attacker_{type}_{suffix}', rouge_res['rouge-l']['f'])
                    self.log_to_all_result(client_id, f'attacker_{type}_{suffix}', rouge_res['rouge-l']['f'])
                    logs[f'attacker_{type}_{suffix}_step'] = rouge_res['rouge-l']['f']
