from typing import Iterator, Any

import torch
from numpy import average
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW

from sfl.config import FLConfig
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.strategy import FLStrategy
from sfl.utils.model import Intermediate, evaluate_accuracy, evaluate_perplexity, get_t5_input, calculate_rouge, \
    dist_corr


class BaseSFLStrategy(FLStrategy):

    def __init__(self, args, fl_config: FLConfig, llm, tokenizer, test_loader=None, sample_batch=None):
        super().__init__(fl_config=fl_config)
        self.args = args
        self.tokenizer = tokenizer
        self.test_loader = test_loader
        self.llm = llm
        self.sample_batch = sample_batch
        self.attack_sample_counter = {}
        self.attack_sample_performs = {}
        self.attack_all_performs = {}

    def client_evaluate(self, global_round, client_id, log):
        if self.task_type == 'classification':
            log['test-acc'] = evaluate_accuracy(self.simulator.llm, self.test_loader)
        elif self.task_type == 'lm':
            ppl = evaluate_perplexity(self.simulator.llm, self.test_loader)
            log['test-ppl'] = ppl

    def client_step(self, client_id: str, global_round, client_epoch, llm: SplitWrapperModel, iterator: Iterator,
                    config: FLConfig):
        optimizer = AdamW(llm.parameters(), lr=config.lr)
        if self.task_type == 'classification':
            optimizer = Adam(llm.parameters(), lr=1e-4)
        avg_loss = 0
        avg_self_rouge = 0
        batch_num = 0
        with tqdm(total=config.client_steps) as pbar:
            for step, batch in enumerate(iterator):
                optimizer.zero_grad()
                if llm.type == 'encoder-decoder':
                    outputs = llm(**get_t5_input(batch, self.tokenizer, llm.device))
                else:
                    input_ids = batch['input_ids'].to(llm.device)
                    attention_mask = batch['attention_mask'].to(llm.device)
                    labels = input_ids
                    if 'labels' in batch and self.llm.task_type != 'lm':
                        labels = batch['labels'].to(llm.device)
                    outputs = llm(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

                loss = outputs.loss
                pbar.set_description(
                    f'Client {client_id} Epoch {client_epoch} Step {self.simulator.get_current_step(client_id, step)} Loss {loss.item():.3f}')
                avg_loss += loss.detach().cpu().item()
                if config.noise_mode in ['dc']:
                    b2tr, tr2t, all_inter = llm.get_all_inter(detach=False)
                    embed = all_inter['embedding']
                    dcor = dist_corr(embed.fx, b2tr.fx)
                    loss += config.noise_beta_dc * dcor
                if config.noise_mode in ['grad', 'both']:
                    loss.backward(retain_graph=True)
                    b2tr, tr2t, all_inter = llm.get_all_inter(detach=False)
                    b2tr.fx.grad = None
                    grad = tr2t.grad.clone()
                    tr2t.fx.grad = None
                    for _, p in llm.get_bottom_params():
                        p.grad = None
                    for _, p in llm.get_trunk_params():
                        p.grad = None
                    noise = torch.randn_like(grad) * ((config.noise_scale_grad * 1.0) ** 2)
                    grad = grad + noise
                    tr2t.fx.backward(grad)
                else:
                    loss.backward()

                batch_num += 1
                optimizer.step()

                if self.task_type == 'lm':
                    avg_self_rouge += \
                        calculate_rouge(self.tokenizer, outputs.logits.argmax(dim=-1), batch['input_text'])['rouge-l'][
                            'f']
                # if self.args.self_pt_enable:
                #     outputs_pt = self.simulator.restored_forward('top', input_ids=input_ids, labels=input_ids,
                #                                                  attention_mask=attention_mask)
                #     avg_self_rouge_pt += \
                #         calculate_rouge(self.tokenizer, outputs_pt.logits, batch['input_text'])['rouge-l']['f']
                self.step_done(client_id, step, batch,
                               {"step_loss": loss.item(), "avg_loss": float(avg_loss / batch_num),
                                "self": float(avg_self_rouge / batch_num)})
                #  "self_pt": float(avg_self_rouge_pt / batch_num)})  # Collect gradients
                pbar.update(1)
                if 0 < config.max_global_step <= self.simulator.get_current_step(client_id, step)[1] + 1:
                    break

    def callback_intermediate_result(self, global_round, client_id, local_epoch, local_step, global_step,
                                     b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                     all_inter: dict[int, Intermediate],
                                     batch, logs
                                     ):
        # 一定频率进行攻击
        self.attack_all_performs.setdefault(client_id, {})
        # 每batch触发攻击
        self.normal_attacker_triggered(global_round, client_id, local_epoch, local_step, b2tr_inter, tr2t_inter,
                                       all_inter, batch, logs)
        if (global_step + 1) % self.args.attacker_freq == 0:
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
        pass

    def sample_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[Any, Intermediate],
                                  batch, logs):
        pass
