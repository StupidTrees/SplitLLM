import argparse
import os
import random
import sys

import torch
import wandb
from torch.optim import Adam
from tqdm import tqdm

sys.path.append(os.path.abspath('../../..'))
from experiments.scripts.py.train_attacker import get_save_path
from sfl.simulator.dataset import MixtureFedDataset
from sfl import config
from sfl.config import FLConfig, DRA_test_label, DRA_train_label
from sfl.model.attacker.dra_attacker import MOEDRAttacker, MOEDRAttackerConfig
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, str2bool
from sfl.utils.model import get_t5_input, get_best_gpu, calc_unshift_loss, set_random_seed, \
    evaluate_attacker_rouge


def evaluate(epc, md, attacker, tok, test_data_loader, save_dir):
    """
    恢复的评价指标选用ROUGE
    :return: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-P, ROUGE-L-R
    """
    md.eval()
    md.change_noise_scale(0)
    attacker.eval()
    dl_len = len(test_data_loader)
    with torch.no_grad():
        rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            if md.type == 'encoder-decoder':
                enc_hidden, dec_hidden = md(**get_t5_input(batch, tok, md.device))
                inter = torch.concat([enc_hidden, dec_hidden], dim=1)
            else:
                input_ids = batch['input_ids'].to(md.device)
                attention_mask = batch['input_att_mask'].to(md.device)
                inter = md(input_ids=input_ids, attention_mask=attention_mask)
            logits = attacker(inter)
            result = evaluate_attacker_rouge(tok, logits, batch)
            rouge_1 += result['rouge-1']['f']
            rouge_2 += result['rouge-2']['f']
            rouge_l_f1 += result['rouge-l']['f']
            rouge_l_p += result['rouge-l']['p']
            rouge_l_r += result['rouge-l']['r']

    print(
        f'Epoch {epc} Test Rouge_l_f1: {rouge_l_f1 / dl_len}')
    p = get_save_path(md, save_dir, args)
    if rouge_l_f1 / dl_len > 0.1 and (epc + 1) % 1 == 0 and args.save_checkpoint:
        attacker.save_pretrained(p + f'epoch_{epc}_rouge_{rouge_l_f1 / dl_len:.4f}')
    if args.log_to_wandb:
        log_dict = {'epoch': epc, 'test_rouge_1': rouge_1 / dl_len, 'test_rouge_2': rouge_2 / dl_len,
                    'test_rouge_l_f1': rouge_l_f1 / dl_len, 'test_rouge_l_p': rouge_l_p / dl_len,
                    'test_rouge_l_r': rouge_l_r / dl_len}
        wandb.log(log_dict)
    md.train(True)
    attacker.train(True)
    return rouge_1 / dl_len, rouge_2 / dl_len, rouge_l_f1 / dl_len, rouge_l_p / dl_len, rouge_l_r / dl_len


def train_attacker(args):
    """
    训练攻击模型
    :param args:
    """

    model, tokenizer = get_model_and_tokenizer(args.model_name, load_bits=args.load_bits)
    if ',' not in args.dataset:
        dataset_cls = get_dataset_class(args.dataset)
        dataset = dataset_cls(tokenizer, [])
        if DRA_train_label[args.dataset] == DRA_test_label[args.dataset]:  # self-testing
            dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                                          type=DRA_train_label[args.dataset],
                                                                          shrink_frac=args.dataset_train_frac,
                                                                          further_test_split=0.3)
        else:
            dataloader = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type=DRA_train_label[args.dataset],
                                                         shrink_frac=args.dataset_train_frac)
            dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                              type=DRA_test_label[args.dataset],
                                                              shrink_frac=args.dataset_test_frac)
    else:
        dataset = MixtureFedDataset(tokenizer, [], args.dataset_train_frac, args.dataset.split(','),
                                    [get_dataset_class(nm) for nm in args.dataset.split(',')])
        dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                                      type=None,
                                                                      shrink_frac=args.dataset_train_frac)

    # if args.attack_model == 'moe2':
    #     dataloader_test = get_dataset('piqa', tokenizer=tokenizer).get_dataloader_unsliced(batch_size=args.batch_size,
    #                                                                                        type='test',
    #                                                                                        shrink_frac=0.01)
    model.config_sfl(FLConfig(collect_intermediates=False,
                              split_point_1=args.split_point_1,
                              split_point_2=args.split_point_2,
                              attack_mode=args.attack_mode,
                              noise_mode=args.noise_mode,
                              noise_scale_dxp=args.noise_scale_dxp,
                              ),
                     param_keeper=None)
    # freeze all parts:
    for name, param in model.named_parameters():
        param.requires_grad = False

    def get_output(text, encoder_model, attack_model):
        t = tokenizer(text, return_tensors="pt")
        inter = encoder_model(t['input_ids'].to(model.device), attention_mask=t['attention_mask'].to(model.device))
        res = attack_model(inter)
        r = tokenizer.decode(res.argmax(dim=-1)[-1], skip_special_tokens=True)
        return r

    # 开始训练Attack Model
    expert_scales = [0, 10.0, 7.5, 5.0]
    if args.attack_model == 'moe2':
        expert_scales = [0] * len(args.dataset.split(','))
    attack_model = MOEDRAttacker(MOEDRAttackerConfig(expert_scales=expert_scales), model.config)
    p = get_save_path(model, args.save_dir, args)
    if os.path.exists(p) and args.skip_exists:
        print('Model exists, skipping...')
        return

    if 'llama2' not in args.model_name:
        device = get_best_gpu()
        model.to(device)

    if args.log_to_wandb:
        wandb.init(project=args.exp_name,
                   name=f"{args.model_name}_{args.split_point_1}",
                   config=vars(args)
                   )

    # 阶段1， 不同专家独立训练解码能力
    attack_model.freeze_parts(experts=False, freeze=True)  # freeze gating
    attack_model.freeze_parts(experts=True, freeze=False)  # freeze gating
    optimizer = Adam([p for p in attack_model.parameters() if p.requires_grad], lr=args.lr)
    attack_model.to(model.device)
    with tqdm(total=args.epochs_expert * len(dataloader)) as pbar:
        for epc in range(args.epochs_expert):
            model.train(True)
            rougeLs = [0] * len(attack_model.config.expert_scales)
            item_count = 0
            item_counts = [0] * len(attack_model.config.expert_scales)
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                if args.attack_model == 'moe':
                    inters = []
                    for noise_scale in attack_model.config.expert_scales:
                        model.change_noise_scale(noise_scale)
                        if model.type == 'encoder-decoder':
                            input_args = get_t5_input(batch, tokenizer, model.device)
                            enc_hidden, dec_hidden = model(**input_args)
                            intermediate = torch.concat([enc_hidden, dec_hidden], dim=1)
                            input_ids = torch.concat([input_args['input_ids'], input_args['decoder_input_ids']],
                                                     dim=1).to(
                                model.device)
                        else:
                            input_ids = batch['input_ids'].to(model.device)
                            attention_mask = batch['input_att_mask'].to(model.device)
                            intermediate = model(input_ids=input_ids, attention_mask=attention_mask)
                        inters.append(intermediate)
                elif args.attack_model == 'moe2':
                    model.change_noise_scale(0)
                    if model.type == 'encoder-decoder':
                        input_args = get_t5_input(batch, tokenizer, model.device)
                        enc_hidden, dec_hidden = model(**input_args)
                        intermediate = torch.concat([enc_hidden, dec_hidden], dim=1)
                        input_ids = torch.concat([input_args['input_ids'], input_args['decoder_input_ids']],
                                                 dim=1).to(
                            model.device)
                    else:
                        input_ids = batch['input_ids'].to(model.device)
                        attention_mask = batch['input_att_mask'].to(model.device)
                        intermediate = model(input_ids=input_ids, attention_mask=attention_mask)
                    inters = [None] * len(attack_model.config.expert_scales)
                    inters[batch['type']] = intermediate
                exp_logits = attack_model.train_exp_forward(inters)
                loss = 0
                for i, logits in enumerate(exp_logits):
                    if logits is None:
                        continue
                    loss += calc_unshift_loss(logits, input_ids)
                    res = evaluate_attacker_rouge(tokenizer, logits, batch)
                    rougeLs[i] += res['rouge-l']['f']
                    item_counts[i] += 1
                loss.backward()
                optimizer.step()

                # 计算训练的ROGUE
                rg_rpt = ", ".join([f"Exp{i}_RgL{rgl / (step + 1):.4f}" for i, rgl in enumerate(rougeLs)])
                if args.attack_model == 'moe2':
                    rg_rpt = ", ".join(
                        [f"Exp{i}_RgL{rgl / (1 if item_counts[i] == 0 else item_counts[i]):.4f}" for i, rgl in
                         enumerate(rougeLs)])
                pbar.set_description(
                    f'EXPERT Epoch {epc} Loss {loss.item():.5f}, {rg_rpt}')
                if step % 300 == 0 and model.type != 'encoder-decoder':
                    q = "To mix food coloring with sugar, you can"
                    print(q, "==>", get_output(q, model, attack_model))
                pbar.update(1)
                item_count += 1
            # # 计算测试集上的ROGUE
            # if (epc + 1) % 5 == 0:
            #     evaluate(epc, model, attack_model, tokenizer, dataloader_test, args.save_dir)
            if args.log_to_wandb:
                log_dict = {'epoch': epc, }
                for i, rgl in enumerate(rougeLs):
                    log_dict.update({f"Expert{i}_RgL": rgl / item_count})
                wandb.log(log_dict)

    # 阶段2，训练Gating
    attack_model.freeze_parts(experts=True, freeze=False)  # activate experts
    attack_model.freeze_parts(experts=False, freeze=False)  # activate gating
    optimizer2 = Adam([p for p in attack_model.parameters() if p.requires_grad], lr=args.lr)
    with tqdm(total=args.epochs_gating * len(dataloader)) as pbar:
        for epc in range(args.epochs_gating):
            model.train(True)
            rougeL_total = 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer2.zero_grad()
                if args.attack_model == 'moe':
                    # 随机噪声规模
                    scales = set(attack_model.config.expert_scales) - {0}
                    numbers = [random.uniform(min(scales), max(scales)) for _ in
                               range(len(attack_model.config.expert_scales))]
                    numbers += [0, 0, max(scales) * 2]
                    model.change_noise_scale(random.choice(numbers))

                if model.type == 'encoder-decoder':
                    input_args = get_t5_input(batch, tokenizer, model.device)
                    enc_hidden, dec_hidden = model(**input_args)
                    intermediate = torch.concat([enc_hidden, dec_hidden], dim=1)
                    input_ids = torch.concat([input_args['input_ids'], input_args['decoder_input_ids']], dim=1).to(
                        model.device)
                else:
                    input_ids = batch['input_ids'].to(model.device)
                    attention_mask = batch['input_att_mask'].to(model.device)
                    intermediate = model(input_ids=input_ids, attention_mask=attention_mask)

                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                res = evaluate_attacker_rouge(tokenizer, logits, batch)
                rougeL_total += res['rouge-l']['f']
                loss.backward()
                optimizer2.step()

                # 计算训练的ROGUE
                pbar.set_description(
                    f'GATING Epoch {epc} Loss {loss.item():.5f}, RougeL {rougeL_total / (step + 1):.4f}')
                if step % 300 == 0 and model.type != 'encoder-decoder':
                    q = "To mix food coloring with sugar, you can"
                    print(q, "==>", get_output(q, model, attack_model))
                pbar.update(1)
                item_count += 1
            # 计算测试集上的ROGUE
            if (epc + 1) % 2 == 0:
                evaluate(epc, model, attack_model, tokenizer, dataloader_test, args.save_dir)
            if args.log_to_wandb:
                log_dict = {'epoch': epc, 'Total_RgL': rougeL_total / item_count}
                wandb.log(log_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='attacker')
    parser.add_argument('--model_download_dir', type=str, default='/root/autodl-tmp/sfl/models')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--save_dir', type=str, default=config.attacker_path)
    parser.add_argument('--dataset', type=str, default='piqa')
    parser.add_argument('--dataset_train_frac', type=float, default=1.0)
    parser.add_argument('--dataset_test_frac', type=float, default=1.0)
    parser.add_argument('--attack_model', type=str, default='moe', help='lstm or ...')
    parser.add_argument('--split_point_1', type=int, default=2, help='split point for b2tr')
    parser.add_argument('--split_point_2', type=int, default=30, help='split point for t2tr')
    parser.add_argument('--attack_mode', type=str, default='tr2t', help='b2tr or t2tr')
    parser.add_argument('--load_bits', type=int, default=8, help='load bits for large models')
    parser.add_argument('--epochs_expert', type=int, default=5)
    parser.add_argument('--epochs_gating', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_mode', type=str, default='dxp')
    parser.add_argument('--noise_scale_dxp', type=float, default=5.0)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)
    parser.add_argument('--log_to_wandb', type=str2bool, default=False)
    parser.add_argument('--skip_exists', type=str2bool, default=True)
    args = parser.parse_args()
    set_random_seed(args.seed)
    train_attacker(args)
