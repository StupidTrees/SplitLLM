import argparse
import os
import sys

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW

sys.path.append(os.path.abspath('../..'))
from sfl.utils.model import calculate_rouge
from sfl import config
import sfl
from sfl.model.split_model import SplitModel
from sfl.simulator.simulator import SFLSimulator
from sfl.simulator.strategy import FLStrategy
from sfl.config import FLConfig
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.utils.training import set_random_seed, get_dataset_class, get_attacker_class, extract_attacker_path


# 定义Client本地学习策略
class QAFLStrategy(FLStrategy):

    def __init__(self, tokenizer, attacker1, attacker2, llm):
        super().__init__()
        self.attacker_rouge_b2tr = []
        self.attacker_rouge_tr2t = []
        self.client_logs = {}
        self.tokenizer = tokenizer
        self.attacker1 = attacker1
        self.attacker2 = attacker2
        self.llm = llm

    def client_step(self, global_round, client_id: str, llm: SplitModel, dataloader: DataLoader, cfg: FLConfig):
        optimizer = AdamW(llm.parameters(), lr=1e-5)
        with tqdm(total=cfg.client_epoch * len(dataloader)) as pbar:
            for epoch in range(cfg.client_epoch):
                avg_loss = 0
                batch_num = 0
                for step, batch in enumerate(dataloader):
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(llm.device)
                    attention_mask = batch['input_att_mask'].to(llm.device)
                    outputs = llm(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
                    self.fp_done(client_id, epoch, step, batch)  # Collect intermediate results
                    loss = outputs.loss
                    pbar.set_description(f'Client {client_id} Epoch {epoch} Loss {loss.item():.3f}')
                    loss.backward()
                    avg_loss += loss.detach().cpu().item()
                    batch_num += 1
                    self.bp_done(client_id, epoch, step, batch)  # Collect gradients
                    optimizer.step()
                    pbar.update(1)
                avg_loss /= batch_num
                self.client_logs.setdefault(client_id, {})
                if len(self.attacker_rouge_b2tr) > 0:
                    avg_rouge1 = sum([r["rouge-l"]["f"] for r in self.attacker_rouge_b2tr]) / len(
                        self.attacker_rouge_b2tr)
                    print(f'ATTACK! Bottom-trunk, Client {client_id} Epoch {epoch} RougeL {avg_rouge1:.3f}')
                    avg_rouge2 = sum([r['rouge-l']['f'] for r in self.attacker_rouge_tr2t]) / len(
                        self.attacker_rouge_tr2t)
                    print(f'ATTACK! Trunk-Top, Client {client_id} Epoch {epoch} RougeL {avg_rouge2:.3f}')
                    self.client_logs[client_id][epoch] = {"bottom-trunk": avg_rouge1, "trunk-top": avg_rouge2,
                                                          "loss": float(avg_loss)}
                else:
                    self.client_logs[client_id][epoch] = {"loss": float(avg_loss)}
                self.attacker_rouge_b2tr.clear()
                self.attacker_rouge_tr2t.clear()

    def aggregation_step(self, global_round, params):
        report = {}
        report['global_round'] = global_round
        for cid, epochs in self.client_logs.items():
            for epc, rep in epochs.items():
                for k, v in rep.items():
                    report[f'client{cid}-epoch{epc}-{k}'] = v
        wandb.log(report)
        print(report)
        self.client_logs = {}
        return super(QAFLStrategy, self).aggregation_step(global_round, params)

    def callback_fp_param(self, global_round, client_id, local_epoch, local_step, b2tr_params, tr2t_params, batch):
        #  这里获取某epoch、step中，前传过程的两次传输参数，b2tr(bottom-trunk), tr2t(trunk-top)
        if client_id == '1' and local_epoch == 1 and (global_round + 1) % 5 == 0 and (local_step + 1) % 10 == 0:
            with torch.no_grad():
                if args.attacker_search == 'True':
                    rouge_res_b2tr = calculate_rouge(self.tokenizer, self.attacker1.search(b2tr_params, self.llm),
                                                     batch['input_text'],
                                                     is_tokens=True)
                    rouge_res_tr2t = calculate_rouge(self.tokenizer, self.attacker2.search(tr2t_params, self.llm),
                                                     batch['input_text'],
                                                     is_tokens=True)
                    self.attacker_rouge_b2tr.append(rouge_res_b2tr)
                    self.attacker_rouge_tr2t.append(rouge_res_tr2t)
                else:
                    rouge_res_b2tr = calculate_rouge(self.tokenizer,
                                                     self.attacker1(b2tr_params.to(self.attacker1.device)),
                                                     batch['input_text'])
                    rouge_res_tr2t = calculate_rouge(self.tokenizer,
                                                     self.attacker2(tr2t_params.to(self.attacker1.device)),
                                                     batch['input_text'])
                    self.attacker_rouge_b2tr.append(rouge_res_b2tr)
                    self.attacker_rouge_tr2t.append(rouge_res_tr2t)
                print(
                    f'ATTACK! Bottom-trunk, Client {client_id} Epoch {local_epoch} Step {local_step} RougeL {rouge_res_b2tr["rouge-l"]["f"]:.3f}')

    def callback_bp_param(self, global_round, client_id, local_epoch, local_step, t2tr_params, tr2b_params, batch):
        #  这里获取某epoch、step中，反传过程的两次传输参数
        pass


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
    attacker_cls = get_attacker_class(args.attack_model)
    attacker_path_1, attacker_path_2 = extract_attacker_path(args)
    attacker = attacker_cls.from_pretrained(attacker_path_1)
    attacker2 = attacker_cls.from_pretrained(attacker_path_2)

    # 配置联邦学习
    client_ids = [str(i) for i in range(3)]
    config = FLConfig(global_round=args.global_round,
                      client_epoch=args.client_epoch,  # 每轮联邦每个Client训x轮
                      split_point_1=args.split_point_1,
                      split_point_2=args.split_point_2,  # [0,1 | 2,3,.... 29| 30, 31]
                      use_lora_at_trunk=args.lora_at_trunk,  # 在trunk部分使用LoRA
                      top_and_bottom_from_scratch=args.client_from_scratch=='True',  # top和bottom都不采用预训练参数.
                      noise_mode=args.noise_mode,
                      noise_scale=args.noise_scale,  # 噪声大小
                      )
    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)

    simulator = SFLSimulator(client_ids=client_ids, strategy=QAFLStrategy(tokenizer, attacker, attacker2, model),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config)
    wandb.init(
        project=args.exp_name,
        name=f"{args.noise_mode}:{args.noise_scale}-prefix=ß{args.attacker_prefix}-search:{args.attacker_search}",
        config=args
    )
    attacker.to(simulator.device)
    attacker2.to(simulator.device)
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='sfl-with-attacker-noise-dxp')
    parser.add_argument('--model_name', type=str, default='gpt2-large')
    parser.add_argument('--dataset', type=str, default='piqa')
    parser.add_argument('--data_shrink_frac', type=float, default=0.15, help='shrink dataset to this fraction')
    parser.add_argument('--attack_model', type=str, default='gru', help='lstm, gru, linear')
    parser.add_argument('--split_point_1', type=int, default=2, help='split point for b2tr')
    parser.add_argument('--split_point_2', type=int, default=30, help='split point for t2tr')
    parser.add_argument('--lora_at_trunk', type=bool, default=True, help='use LoRA at trunk part')
    parser.add_argument('--noise_mode', type=str, default='dxp')
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--attacker_train_label', type=str, default='train')
    parser.add_argument('--attacker_train_frac', type=float, default=1.0)
    parser.add_argument('--attacker_prefix', type=str, default='normal')
    parser.add_argument('--attacker_search', type=str, default='False')
    parser.add_argument('--attacker_dataset', type=str,
                        default='piqa')
    parser.add_argument('--attacker_path', type=str,
                        default=config.attacker_path,
                        help='trained attacker model for b2tr')
    parser.add_argument('--attack_mode', type=str, default='b2tr', help='b2tr or t2tr')
    parser.add_argument('--global_round', type=int, default=30)
    parser.add_argument('--client_from_scratch', type=str, default='False')
    parser.add_argument('--client_epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
