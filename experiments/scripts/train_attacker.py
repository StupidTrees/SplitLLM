import argparse
import os
import sys

import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim import Adam

sys.path.append(os.path.abspath('../..'))
from sfl.config import AttackerConfig
from sfl.model.gpt2.gpt2_split import GPT2SplitLMHeadModel
from sfl.utils import get_best_gpu, set_random_seed
from sfl.simulator.dataset import PIQAFedDataset, GSM8KFedDataset
from sfl.utils import calculate_rouge

from sfl.model.attack_model import LSTMAttackerConfig, LSTMAttackModel, LinearAttackModel, \
    TransformerEncoderAttackModel, TransformerAttackerConfig, TransformerDecoderAttackModel, GRUAttackModel
from sfl.utils import calc_unshift_loss


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
    path = save_dir + f'/attacker/{md.config.name_or_path}/{args.dataset}/{args.attack_model}/{md.fl_config.attack_mode}-{md.fl_config.split_point_1 if md.fl_config.attack_mode == "b2tr" else md.fl_config.split_point_2}/'
    if rouge_l_f1 / dl_len > 0.7:
        attacker.save_pretrained(path + f'epoch_{epc}_rouge_{rouge_l_f1 / dl_len:.4f}')
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
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large", cache_dir=args.model_cache_dir)
    model: GPT2SplitLMHeadModel = GPT2SplitLMHeadModel.from_pretrained("gpt2-large", cache_dir=args.model_cache_dir)
    tokenizer.pad_token_id = model.config.eos_token_id
    device = get_best_gpu()
    model.to(device)

    dataset = PIQAFedDataset(tokenizer, [])
    dataloader = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type='validation')
    dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type='test')
    if args.dataset == 'gsm8k':
        dataset = GSM8KFedDataset(tokenizer, [])
        dataloader = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type='test')
        dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type='train', shrink_frac=0.06)

    from sfl.config import FLConfig
    model.config_sfl(FLConfig(collect_intermediates=False,
                              split_point_1=args.split_point_1,
                              split_point_2=args.split_point_2,
                              attack_mode=args.attack_mode,
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
            evaluate(epc, model, attack_model, tokenizer, dataloader_test, args.save_dir)
            wandb.log({'epoch': epc, 'train_rouge_1': rouge_1, 'train_rouge_2': rouge_2, 'train_rouge_l_f1': rouge_l_f1,
                       'train_rouge_l_p': rouge_l_p, 'train_rouge_l_r': rouge_l_r})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cache_dir', type=str, default='/root/autodl-tmp/sfl/models')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2-large')
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/sfl/models/checkpoints')
    parser.add_argument('--dataset', type=str, default='piqa')
    parser.add_argument('--attack_model', type=str, default='trans-enc', help='lstm or ...')
    parser.add_argument('--split_point_1', type=int, default=2, help='split point for b2tr')
    parser.add_argument('--split_point_2', type=int, default=30, help='split point for t2tr')
    parser.add_argument('--attack_mode', type=str, default='b2tr', help='b2tr or t2tr')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_random_seed(args.seed)
    # convert namespace to dict
    wandb.init(project="sfl-attacker",
               name=f"{args.attack_model}_{args.attack_mode}_{args.dataset}",
               config=vars(args)
               )
    train_attacker(args)
