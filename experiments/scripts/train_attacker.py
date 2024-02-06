import argparse
import os
import sys

import torch
import wandb
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.abspath('../..'))
from sfl import config
from sfl.config import AttackerConfig
from sfl.config import FLConfig
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.utils.training import get_best_gpu, set_random_seed, calc_unshift_loss, get_dataset_class, calc_shifted_loss
from sfl.utils.model import calculate_rouge
from sfl.model.attack_model import LSTMAttackerConfig, LSTMAttackModel, LinearAttackModel, \
    TransformerEncoderAttackModel, TransformerAttackerConfig, TransformerDecoderAttackModel, GRUAttackModel


def evaluate(epc, md, attacker, tok, test_data_loader, save_dir):
    """
    恢复的评价指标选用ROUGE
    :return: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-P, ROUGE-L-R
    """
    md.eval()
    attacker.eval()
    dl_len = len(test_data_loader)
    with torch.no_grad():
        rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            input_ids = batch['input_ids'].to(md.device)
            attention_mask = batch['input_att_mask'].to(md.device)
            inter = md(input_ids=input_ids, attention_mask=attention_mask)
            logits = attacker(inter)
            result = calculate_rouge(tok, logits, batch['input_text'])
            rouge_1 += result['rouge-1']['f']
            rouge_2 += result['rouge-2']['f']
            rouge_l_f1 += result['rouge-l']['f']
            rouge_l_p += result['rouge-l']['p']
            rouge_l_r += result['rouge-l']['r']
    print(
        f'Epoch {epc} Test Rouge_1: {rouge_1 / dl_len}, Rouge_2: {rouge_2 / dl_len}, Rouge_l_f1: {rouge_l_f1 / dl_len}, Rouge_l_p: {rouge_l_p / dl_len}, Rouge_l_r: {rouge_l_r / dl_len}')
    p = os.path.join(save_dir,
                     f'{attacker.config.target_model}/{args.dataset}/{args.dataset_train_label}*{args.dataset_train_frac:.3f}-{args.dataset_test_label}*{args.dataset_test_frac:.3f}'
                     f'/{args.attack_model}/{md.fl_config.attack_mode}-{md.fl_config.split_point_1 if md.fl_config.attack_mode == "b2tr" else md.fl_config.split_point_2}/')
    attacker_prefix = 'normal/'
    if md.fl_config.noise_mode != 'none':
        attacker_prefix = f'{md.fl_config.noise_mode}:{md.fl_config.noise_scale}/'
    p += attacker_prefix
    if rouge_l_f1 / dl_len > 0.1 and (epc + 1) % 10 == 0 and args.save_checkpoint:
        attacker.save_pretrained(p + f'epoch_{epc}_rouge_{rouge_l_f1 / dl_len:.4f}')
    if args.log_to_wandb == 'True':
        wandb.log({'epoch': epc, 'test_rouge_1': rouge_1 / dl_len, 'test_rouge_2': rouge_2 / dl_len,
                   'test_rouge_l_f1': rouge_l_f1 / dl_len, 'test_rouge_l_p': rouge_l_p / dl_len,
                   'test_rouge_l_r': rouge_l_r / dl_len})
    md.train(True)
    attacker.train(True)
    return rouge_1 / dl_len, rouge_2 / dl_len, rouge_l_f1 / dl_len, rouge_l_p / dl_len, rouge_l_r / dl_len


def train_attacker(args):
    """
    训练攻击模型
    :param args:
    """

    tokenizer = AutoTokenizer.from_pretrained(args.model_download_dir + f"/{args.model_name}/")
    model: GPT2SplitLMHeadModel = GPT2SplitLMHeadModel.from_pretrained(args.model_download_dir + f"/{args.model_name}/")
    tokenizer.pad_token_id = model.config.eos_token_id
    dataset_cls = get_dataset_class(args.dataset)
    dataset = dataset_cls(tokenizer, [])

    if args.dataset_test_label == args.dataset_train_label:  # self-testing
        dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                                      type=args.dataset_train_label,
                                                                      shrink_frac=args.dataset_train_frac,
                                                                      further_test_split=0.3)
    else:
        dataloader = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type=args.dataset_train_label,
                                                     shrink_frac=args.dataset_train_frac)
        dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type=args.dataset_test_label,
                                                          shrink_frac=args.dataset_test_frac)
    device = get_best_gpu()
    model.to(device)
    model.config_sfl(FLConfig(collect_intermediates=False,
                              split_point_1=args.split_point_1,
                              split_point_2=args.split_point_2,
                              attack_mode=args.attack_mode,
                              noise_mode=args.noise_mode,
                              noise_scale=args.noise_scale,
                              ),
                     param_keeper=None)
    # freeze all parts:
    for name, param in model.named_parameters():
        param.requires_grad = False

    def get_output(text, encoder_model, attack_model):
        t = tokenizer(text, return_tensors="pt")
        inter = encoder_model(t['input_ids'].to(device), attention_mask=t['attention_mask'].to(device))
        res = attack_model(inter)
        r = tokenizer.decode(res.argmax(dim=-1)[-1], skip_special_tokens=True)
        return r

    # 开始训练Attack Model
    attack_model = LSTMAttackModel(LSTMAttackerConfig(), model.config)
    if args.attack_model == 'lstm':
        attack_model = LSTMAttackModel(LSTMAttackerConfig(), model.config)
    elif args.attack_model == 'gru':
        attack_model = GRUAttackModel(LSTMAttackerConfig(), model.config)
    elif args.attack_model == 'linear':
        attack_model = LinearAttackModel(AttackerConfig(), model.config)
    elif args.attack_model == 'trans-enc':
        attack_model = TransformerEncoderAttackModel(TransformerAttackerConfig(), model.config)
    elif args.attack_model == 'trans-dec':
        attack_model = TransformerDecoderAttackModel(TransformerAttackerConfig(), model.config)

    optimizer = Adam(attack_model.parameters(), lr=args.lr)
    model.to(device)
    attack_model.to(device)
    epoch = args.epochs
    if args.log_to_wandb == 'True':
        wandb.init(project="attacker-different-sp",
                   name=f"{args.model_name}_{args.split_point_1}",
                   config=vars(args)
                   )

    evaluate(0, model, attack_model, tokenizer, dataloader_test, args.save_dir)
    with tqdm(total=epoch * len(dataloader)) as pbar:
        for epc in range(epoch):
            model.train(True)
            rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['input_att_mask'].to(device)
                intermediate = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                loss.backward()
                optimizer.step()
                # 计算训练的ROGUE
                res = calculate_rouge(tokenizer, logits, batch['input_text'])
                rouge_1 += res['rouge-1']['f']
                rouge_2 += res['rouge-2']['f']
                rouge_l_f1 += res['rouge-l']['f']
                rouge_l_p += res['rouge-l']['p']
                rouge_l_r += res['rouge-l']['r']
                pbar.set_description(f'Epoch {epc} Loss {loss.item():.5f}, Rouge_Lf1 {rouge_l_f1 / (step + 1):.4f}')
                if step % 300 == 0:
                    q = "To mix food coloring with sugar, you can"
                    print(q, "==>", get_output(q, model, attack_model))
                pbar.update(1)
            rouge_1 /= len(dataloader)
            rouge_2 /= len(dataloader)
            rouge_l_f1 /= len(dataloader)
            rouge_l_p /= len(dataloader)
            rouge_l_r /= len(dataloader)
            print(
                f'Epoch {epc} Train Rouge_1: {rouge_1}, Rouge_2: {rouge_2}, Rouge_l_f1: {rouge_l_f1}, Rouge_l_p: {rouge_l_p}, Rouge_l_r: {rouge_l_r}')
            # 计算测试集上的ROGUE
            if (epc + 1) % 5 == 0:
                evaluate(epc, model, attack_model, tokenizer, dataloader_test, args.save_dir)
            if args.log_to_wandb == 'True':
                wandb.log(
                    {'epoch': epc, 'train_rouge_1': rouge_1, 'train_rouge_2': rouge_2, 'train_rouge_l_f1': rouge_l_f1,
                     'train_rouge_l_p': rouge_l_p, 'train_rouge_l_r': rouge_l_r})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_download_dir', type=str, default='/root/autodl-tmp/sfl/models')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--save_dir', type=str, default=config.attacker_path)
    parser.add_argument('--dataset', type=str, default='piqa')
    parser.add_argument('--dataset_train_label', type=str, default='test')
    parser.add_argument('--dataset_test_label', type=str, default='test')
    parser.add_argument('--dataset_train_frac', type=float, default=1.0)
    parser.add_argument('--dataset_test_frac', type=float, default=1.0)
    parser.add_argument('--attack_model', type=str, default='gru', help='lstm or ...')
    parser.add_argument('--split_point_1', type=int, default=2, help='split point for b2tr')
    parser.add_argument('--split_point_2', type=int, default=30, help='split point for t2tr')
    parser.add_argument('--attack_mode', type=str, default='tr2t', help='b2tr or t2tr')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_mode', type=str, default='none')
    parser.add_argument('--noise_scale', type=float, default=5.0)
    parser.add_argument('--save_checkpoint', type=str, default='False')
    parser.add_argument('--log_to_wandb', type=str, default='False')
    args = parser.parse_args()
    set_random_seed(args.seed)
    train_attacker(args)
