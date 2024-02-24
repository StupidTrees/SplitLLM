import argparse
import os
import sys

import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW

sys.path.append(os.path.abspath('../..'))
from sfl.model.dlgattacker import GPT2TopDLGAttacker
from experiments.scripts.basic_strategy import BaseSFLStrategy
from sfl.utils.experiments import add_sfl_params, str2bool
from sfl.utils.model import calculate_rouge
import sfl
from sfl.simulator.simulator import SFLSimulator
from sfl.config import FLConfig, Intermediate
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.utils.training import set_random_seed, get_dataset_class, get_attacker_class, extract_attacker_path, \
    evaluate_perplexity


# 定义Client本地学习策略
class MirrorFLStrategy(BaseSFLStrategy):

    def __init__(self, args, tokenizer, attacker1, attacker2, llm, test_loader):
        super().__init__(args, tokenizer, attacker1, attacker2, llm, test_loader)
        self.attacked_samples = []

    def client_evaluate(self, global_round, client_id, log):
        ppl = evaluate_perplexity(self.simulator.llm, self.test_loader)
        log['test-ppl'] = ppl

        def non_mirror_test():
            ppl = evaluate_perplexity(self.simulator.llm, self.test_loader)
            log['no-mirror-test-ppl'] = ppl

        self.simulator.restored_run(non_mirror_test, parts=['top'], key='pretrained', write_back=False)

        llm = self.simulator.llm
        param_frozen = []
        # freeze bottom and trunk parts
        for nm, p in llm.get_bottom_params():
            param_frozen.append(p)
            p.requires_grad = False
        for nm, p in llm.get_trunk_params():
            param_frozen.append(p)
            p.requires_grad = False
        bk_ci = llm.fl_config.collect_intermediates
        llm.fl_config.collect_intermediates = False

        def mirror_ft():
            top_params = [x for _, x in llm.get_top_params()]
            optimizer = AdamW(top_params, lr=1e-5)
            batch_num = 0
            with tqdm(total=len(self.attacked_samples)) as pbar:
                for step, batch in enumerate(self.attacked_samples):
                    batch = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(llm.device)
                    attention_mask = batch['attention_mask'].to(llm.device)
                    outputs = llm(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
                    loss = outputs.loss
                    pbar.set_description(f'Mirror FT Loss {loss.item():.3f}')
                    loss.backward()
                    batch_num += 1
                    optimizer.step()
                    pbar.update(1)
            ppl = evaluate_perplexity(llm, self.test_loader)
            log['mirror-test-ppl'] = ppl

        self.simulator.restored_run(mirror_ft, parts=['top'], key='mirror', write_back=True)
        # restore all the params
        for param in param_frozen:
            param.requires_grad = True
        llm.fl_config.collect_intermediates = bk_ci
        self.attacked_samples.clear()

    def sample_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[int, Intermediate],
                                  batch, logs):
        with torch.no_grad():
            for type, atk in zip(['tr2t', 'b2tr'], [self.attacker1, self.attacker2]):
                if atk is None:
                    continue
                atk.to(self.simulator.device)
                inter = b2tr_inter if type == 'b2tr' else tr2t_inter
                attacked = atk(inter.fx.to(atk.device))
                rouge_res = calculate_rouge(self.tokenizer, attacked, batch['input_text'])
                self.log_to_sample_result(client_id, f'attacker_{type}', rouge_res['rouge-l']['f'])
                self.log_to_all_result(client_id, f'attacker_{type}', rouge_res['rouge-l']['f'])
                logs[f'attacker_{type}_step'] = rouge_res['rouge-l']['f']
        gt_init = None
        if args.dlg_init_with_dra:
            gt_init = attacked
        gt = self.dlg.fit(tr2t_inter.fx.to(self.simulator.device), tr2t_inter.grad.to(self.simulator.device),
                          epochs=self.args.dlg_epochs,
                          adjust=args.dlg_adjust,
                          beta=self.args.dlg_beta,
                          gt_init=gt_init,
                          gt_reg=self.args.dlg_dra_reg,
                          temp_range=self.args.dlg_temp_range
                          )
        rouge = calculate_rouge(self.tokenizer, gt, batch['input_text'])
        self.log_to_sample_result(client_id, 'tag_rouge_lf', rouge['rouge-l']['f'])
        self.log_to_all_result(client_id, 'tag_rouge_lf', rouge['rouge-l']['f'])
        output_texts = [self.tokenizer.decode(gt.argmax(dim=-1)[i], skip_special_tokens=True) for i in
                        range(len(gt))]
        self.attacked_samples.append(output_texts)


def sfl_with_attacker(args):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    model = GPT2SplitLMHeadModel.from_pretrained(os.path.join(sfl.config.model_download_dir, args.model_name))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 50256
    # 加载攻击模型
    attacker, attacker2 = extract_attacker_path(args, get_attacker_class(args.attacker_model))

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
                      dataset_type=args.dataset_label,
                      collect_all_layers=args.collect_all_layers,
                      )

    # 加载TAG攻击模型
    mocker = None
    if args.dlg_enable:
        # !需要在LoRA加上去之前进行复制
        mocker = GPT2TopDLGAttacker(config, model)
    # 加载数据集
    dataset_cls = get_dataset_class(args.dataset)
    fed_dataset = dataset_cls(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=args.data_shrink_frac)
    test_dataset = dataset_cls(tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=MirrorFLStrategy(args, tokenizer, attacker, attacker2, model, test_loader),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config)
    wandb.init(
        project=args.exp_name,
        name=f"T-freq[{args.attacker_freq}]",
        config=args
    )
    mocker.to(simulator.device)
    simulator.strategy.dlg = mocker
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    parser.add_argument('--dlg_init_with_dra', type=str2bool, default=False,
                        help='initialize GT vector with DRA attacker')
    parser.add_argument('--dlg_dra_reg', type=float, default=0.0,
                        help='Add regularization term to make GT closer to DRA result')
    parser.add_argument('--dlg_temp_range', type=float, default=0.5)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
