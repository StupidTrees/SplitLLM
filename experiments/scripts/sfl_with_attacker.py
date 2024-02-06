import argparse
import os
import sys
from typing import Iterator

import torch
import wandb
from numpy import average
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW

sys.path.append(os.path.abspath('../..'))
from sfl.model.mocker import GPT2TopMocker
from sfl.utils.model import calculate_rouge
from sfl.simulator.strategy import FLStrategy
from sfl import config
import sfl
from sfl.model.split_model import SplitModel
from sfl.simulator.simulator import SFLSimulator
from sfl.config import FLConfig
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.utils.training import set_random_seed, get_dataset_class, get_attacker_class, extract_attacker_path, \
    evaluate_perplexity, str2bool


# 定义Client本地学习策略
class QAFLStrategy(FLStrategy):

    def __init__(self, tokenizer, attacker1, attacker2, llm, test_loader):
        super().__init__()
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
        ppl = evaluate_perplexity(self.simulator.llm, self.test_loader)
        log['test-ppl'] = ppl

    def client_step(self, client_id: str, global_round, client_epoch, llm: SplitModel, iterator: Iterator,
                    config: FLConfig):
        optimizer = AdamW(llm.parameters(), lr=1e-5)
        avg_loss = 0
        avg_self_rouge = 0
        avg_self_rouge_pt = 0
        batch_num = 0
        with tqdm(total=config.client_steps) as pbar:
            for step, batch in enumerate(iterator):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(llm.device)
                attention_mask = batch['input_att_mask'].to(llm.device)
                outputs = llm(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs.loss
                pbar.set_description(
                    f'Client {client_id} Epoch {client_epoch} Step {self.simulator.get_current_step(client_id, step)} Loss {loss.item():.3f}')
                loss.backward()
                batch_num += 1
                optimizer.step()
                avg_loss += loss.detach().cpu().item()
                avg_self_rouge += \
                    calculate_rouge(self.tokenizer, outputs.logits.argmax(dim=-1), batch['input_text'])['rouge-l'][
                        'f']
                if args.self_pt_enable == 'True':
                    outputs_pt = self.simulator.restored_forward('top', input_ids=input_ids, labels=input_ids,
                                                                 attention_mask=attention_mask)
                    avg_self_rouge_pt += \
                        calculate_rouge(self.tokenizer, outputs_pt.logits, batch['input_text'])['rouge-l']['f']
                self.step_done(client_id, step, batch,
                               {"loss": float(avg_loss / batch_num), "self": float(avg_self_rouge / batch_num),
                                "self_pt": float(avg_self_rouge_pt / batch_num)})  # Collect gradients
                pbar.update(1)

    def callback_intermediate_result(self, global_round, client_id, local_epoch, local_step, b2tr_fx, tr2b_grad,
                                     tr2t_fx, t2tr_grad, batch, logs):
        #  这里获取某epoch、step中，反传过程的两次传输参数

        # 一定频率进行攻击
        self.attack_all_performs.setdefault(client_id, {})
        if (local_step + 1) % args.attacker_freq == 0:
            self.attack_sample_counter[client_id] = 0
            self.attack_sample_performs[client_id] = {}

        if client_id in self.attack_sample_counter:
            self.attack_sample_counter[client_id] += 1
            if self.dlg is not None:
                gt = self.dlg.fit(tr2t_fx.to(self.simulator.device), t2tr_grad, epochs=args.dlg_epochs,
                                  adjust=args.dlg_adjust, beta=args.dlg_beta)
                rouge = calculate_rouge(self.tokenizer, gt, batch['input_text'])
                self.attack_sample_performs[client_id].setdefault('tag_rouge_lf', [])
                self.attack_all_performs[client_id].setdefault('tag_rouge_lf', [])
                self.attack_sample_performs[client_id]['tag_rouge_lf'].append(rouge['rouge-l']['f'])
                self.attack_all_performs[client_id]['tag_rouge_lf'].append(rouge['rouge-l']['f'])
            rg = [False]
            if args.attacker_search == 'True':
                rg.append(True)
            for search in rg:
                suffix = f'{"search" if search else "normal"}'
                with torch.no_grad():
                    if self.attacker1 is not None:
                        self.attacker1.to(self.simulator.device)
                        if search:
                            attacked_b2tr = self.attacker1.search(b2tr_fx, self.llm)
                        else:
                            attacked_b2tr = self.attacker1(b2tr_fx.to(self.attacker1.device))
                        rouge_res_b2tr = calculate_rouge(self.tokenizer, attacked_b2tr, batch['input_text'])
                        self.attack_sample_performs[client_id].setdefault(f'attacker_b2tr_{suffix}', [])
                        self.attack_all_performs[client_id].setdefault(f'attacker_b2tr_{suffix}', [])
                        self.attack_sample_performs[client_id][f'attacker_b2tr_{suffix}'].append(
                            rouge_res_b2tr['rouge-l']['f'])
                        self.attack_all_performs[client_id][f'attacker_b2tr_{suffix}'].append(
                            rouge_res_b2tr['rouge-l']['f'])
                        logs[f'attacker_b2tr_{suffix}_step'] = rouge_res_b2tr['rouge-l']['f']
                    if self.attacker2 is not None:
                        self.attacker2.to(self.simulator.device)
                        if search:
                            attacked_tr2t = self.attacker2.search(tr2t_fx, self.llm)
                        else:
                            attacked_tr2t = self.attacker2(tr2t_fx.to(self.attacker1.device))
                        rouge_res_tr2t = calculate_rouge(self.tokenizer, attacked_tr2t, batch['input_text'])
                        self.attack_sample_performs[client_id].setdefault(f'attacker_tr2t_{suffix}', [])
                        self.attack_all_performs[client_id].setdefault(f'attacker_tr2t_{suffix}', [])
                        self.attack_sample_performs[client_id][f'attacker_tr2t_{suffix}'].append(
                            rouge_res_tr2t['rouge-l']['f'])
                        self.attack_all_performs[client_id][f'attacker_tr2t_{suffix}'].append(
                            rouge_res_tr2t['rouge-l']['f'])
                        logs[f'attacker_tr2t_{suffix}_step'] = rouge_res_tr2t['rouge-l']['f']

            for key, list in self.attack_all_performs[client_id].items():
                logs[key + '_avg'] = average(list)
            if self.attack_sample_counter[client_id] >= args.attacker_samples:
                del self.attack_sample_counter[client_id]
                for key, list in self.attack_sample_performs[client_id].items():
                    logs[key + '_sampled'] = average(list)
                del self.attack_sample_performs[client_id]


# 测试模型输出
def get_output(text, tokenizer, model):
    t = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    res = model(t['input_ids'].to(model.device), attention_mask=t['attention_mask'].to(model.device))
    r = tokenizer.decode(res.logits.argmax(dim=-1)[-1], skip_special_tokens=True)
    return r


def sfl_with_attacker(args):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    model = GPT2SplitLMHeadModel.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 50256
    # 加载攻击模型
    attacker1, attacker2 = extract_attacker_path(args, get_attacker_class(args.attacker_model))

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = FLConfig(global_round=args.global_round,
                      client_evaluate_freq=args.evaluate_freq,
                      client_steps=args.client_steps,
                      client_epoch=args.client_epoch,  # 每轮联邦每个Client训x轮
                      split_point_1=int(args.split_points.split('-')[0]),
                      split_point_2=int(args.split_points.split('-')[1]),
                      use_lora_at_trunk=args.lora_at_trunk,  # 在trunk部分使用LoRA
                      use_lora_at_top=args.lora_at_top,
                      use_lora_at_bottom=args.lora_at_bottom,
                      top_and_bottom_from_scratch=args.client_from_scratch,  # top和bottom都不采用预训练参数.
                      noise_mode=args.noise_mode,
                      noise_scale=args.noise_scale,  # 噪声大小
                      collect_intermediates=True,
                      )
    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    test_dataset = dataset_cls(tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=QAFLStrategy(tokenizer, attacker1, attacker2, model, test_loader),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config)
    wandb.init(
        project=args.exp_name,
        name=f"{args.split_points}",
        config=args
    )
    # 加载TAG攻击模型
    mocker = None
    if args.dlg_enable == 'True':
        # model.print_split_model()
        mocker = GPT2TopMocker(config, model)
        mocker.to(simulator.device)
    simulator.strategy.dlg = mocker
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='sfl-watt-dxp')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--data_shrink_frac', type=float, default=0.15, help='shrink dataset to this fraction')
    parser.add_argument('--test_data_shrink_frac', type=float, default=0.15, help='shrink dataset to this fraction')
    parser.add_argument('--evaluate_freq', type=int, default=25)
    parser.add_argument('--split_points', type=str, default='2-10', help='split points, b2tr-tr2t')
    parser.add_argument('--lora_at_trunk', type=str2bool, default=True, help='use LoRA at trunk part')
    parser.add_argument('--lora_at_bottom', type=str2bool, default=False, help='use LoRA at bottom part')
    parser.add_argument('--lora_at_top', type=str2bool, default=False, help='use LoRA at top part')
    parser.add_argument('--noise_mode', type=str, default='dxp')
    parser.add_argument('--noise_scale', type=float, default=5.0)
    parser.add_argument('--attacker_model', type=str, default='gru', help='lstm, gru, linear')
    parser.add_argument('--attacker_train_label', type=str, default='validation')
    parser.add_argument('--attacker_train_frac', type=float, default=1.0)
    parser.add_argument('--attacker_prefix', type=str, default='normal')
    parser.add_argument('--attacker_search', type=str, default='False')
    parser.add_argument('--attacker_freq', type=int, default=50, help='attack every * steps')
    parser.add_argument('--attacker_samples', type=int, default=10, help='attack how many batches each time')
    parser.add_argument('--attacker_dataset', type=str,
                        default='wikitext')
    parser.add_argument('--attacker_path', type=str,
                        default=config.attacker_path,
                        help='trained attacker model for b2tr')
    parser.add_argument('--attacker_b2tr_enable', type=str, default='True')
    parser.add_argument('--attacker_b2tr_sp', type=int, default=-1)
    parser.add_argument('--attacker_tr2t_enable', type=str, default='True')
    parser.add_argument('--attacker_tr2t_sp', type=int, default=-1)
    parser.add_argument('--client_num', type=int, default=3)
    parser.add_argument('--global_round', type=int, default=4)
    parser.add_argument('--client_from_scratch', type=str, default='False')
    parser.add_argument('--dlg_enable', type=str, default='True')
    parser.add_argument('--dlg_epochs', type=int, default=300)
    parser.add_argument('--dlg_adjust', type=int, default=0)
    parser.add_argument('--dlg_beta', type=float, default=0.5)
    parser.add_argument('--self_pt_enable', type=str, default="False")
    parser.add_argument('--client_steps', type=int, default=50)
    parser.add_argument('--client_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
