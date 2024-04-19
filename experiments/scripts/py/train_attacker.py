import argparse
import os
import random
import sys

import torch
import wandb
from torch.optim import Adam, AdamW
from tqdm import tqdm

sys.path.append(os.path.abspath('../../..'))
from sfl.simulator.dataset import MixtureFedDataset
import sfl
from sfl.config import FLConfig, DRAttackerConfig, dxp_moe_range, gaussian_moe_range
from sfl.model.attacker.dra_attacker import LSTMDRAttacker, GRUDRAttacker, LinearDRAttacker, LSTMDRAttackerConfig, \
    TransformerDRAttackerConfig, DecoderDRAttacker, AttnGRUDRAttacker, TransformerGRUDRAttackerConfig, \
    GRUAttnDRAttacker, AttnDRAttacker
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, add_train_dra_params, get_tokenizer
from sfl.utils.model import get_t5_input, get_best_gpu, calc_unshift_loss, set_random_seed, \
    evaluate_attacker_rouge, random_choose_noise

from sfl.config import DRA_train_label, DRA_test_label


def get_save_path(fl_config, save_dir, args):
    model_name = args.model_name
    cut_layer = fl_config.split_point_1 if fl_config.attack_mode == "b2tr" else fl_config.split_point_2
    if 'llama' in model_name:
        model_name += f'-{args.load_bits}bits'
    if ',' in args.dataset:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/Tr{args.dataset_train_frac:.3f}-Ts*{args.dataset_test_frac:.3f}'
                         f'/{args.attack_model}/layer{cut_layer}/')
    else:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/{DRA_train_label[args.dataset]}*{args.dataset_train_frac:.3f}-{DRA_test_label[args.dataset]}*{args.dataset_test_frac:.3f}'
                         f'/{args.attack_model}/layer{cut_layer}/')
    attacker_prefix = 'normal/'
    if fl_config.noise_mode == 'dxp':
        attacker_prefix = f'{fl_config.noise_mode}:{fl_config.noise_scale_dxp}/'
    elif fl_config.noise_mode == 'gaussian':
        attacker_prefix = f'{fl_config.noise_mode}:{fl_config.noise_scale_gaussian}/'
    elif fl_config.noise_mode == 'mix':
        attacker_prefix = 'mix/'
    if 'moe' in args.attack_model:
        attacker_prefix = f'{fl_config.noise_mode}/'
    if args.require_prefix is not None:
        attacker_prefix = f'{args.require_prefix}/'
    p += attacker_prefix
    return p


def evaluate(epc, md, attacker, tok, test_data_loader, args):
    """
    恢复的评价指标选用ROUGE
    :return: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-P, ROUGE-L-R
    """
    md.eval()
    if args.noise_mode == 'mix' or args.noise_scale_dxp < 0 or args.noise_scale_gaussian < 0:
        md.change_noise(0)
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
            result, _, _ = evaluate_attacker_rouge(tok, logits, batch)
            rouge_1 += result['rouge-1']['f']
            rouge_2 += result['rouge-2']['f']
            rouge_l_f1 += result['rouge-l']['f']
            rouge_l_p += result['rouge-l']['p']
            rouge_l_r += result['rouge-l']['r']

    print(
        f'Epoch {epc} Test Rouge_l_f1: {rouge_l_f1 / dl_len}')  # , Test2 Rouge_l_f1: {rouge_l_f1_x / dl_len if attacker2 else 0}')
    p = get_save_path(md.fl_config, args.save_dir, args)
    if rouge_l_f1 / dl_len > 0.1 and args.save_checkpoint:
        attacker.save_pretrained(p + f'epoch_{epc}_rouge_{rouge_l_f1 / dl_len:.4f}')
    if args.log_to_wandb:
        log_dict = {'epoch': epc, 'test_rouge_1': rouge_1 / dl_len, 'test_rouge_2': rouge_2 / dl_len,
                    'test_rouge_l_f1': rouge_l_f1 / dl_len, 'test_rouge_l_p': rouge_l_p / dl_len,
                    'test_rouge_l_r': rouge_l_r / dl_len}
        wandb.log(log_dict)
    md.train(True)
    attacker.train(True)
    return rouge_1 / dl_len, rouge_2 / dl_len, rouge_l_f1 / dl_len, rouge_l_p / dl_len, rouge_l_r / dl_len


# def evaluate(epc, md, attacker, tok, test_data_loader, save_dir):
#     """
#     恢复的评价指标选用ROUGE
#     :return: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-P, ROUGE-L-R
#     """
#     md.eval()
#     md.change_noise_scale(0)
#     attacker.eval()
#     dl_len = len(test_data_loader)
#     with torch.no_grad():
#         rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
#         for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
#             if md.type == 'encoder-decoder':
#                 enc_hidden, dec_hidden = md(**get_t5_input(batch, tok, md.device))
#                 inter = torch.concat([enc_hidden, dec_hidden], dim=1)
#             else:
#                 input_ids = batch['input_ids'].to(md.device)
#                 attention_mask = batch['input_att_mask'].to(md.device)
#                 inter = md(input_ids=input_ids, attention_mask=attention_mask)
#             logits = attacker(inter)
#             result = evaluate_attacker_rouge(tok, logits, batch)
#             rouge_1 += result['rouge-1']['f']
#             rouge_2 += result['rouge-2']['f']
#             rouge_l_f1 += result['rouge-l']['f']
#             rouge_l_p += result['rouge-l']['p']
#             rouge_l_r += result['rouge-l']['r']
#
#     print(
#         f'Epoch {epc} Test Rouge_l_f1: {rouge_l_f1 / dl_len}')
#     p = get_save_path(md, save_dir, args)
#     if rouge_l_f1 / dl_len > 0.1 and (epc + 1) % 1 == 0 and args.save_checkpoint:
#         attacker.save_pretrained(p + f'epoch_{epc}_rouge_{rouge_l_f1 / dl_len:.4f}')
#     if args.log_to_wandb:
#         log_dict = {'epoch': epc, 'test_rouge_1': rouge_1 / dl_len, 'test_rouge_2': rouge_2 / dl_len,
#                     'test_rouge_l_f1': rouge_l_f1 / dl_len, 'test_rouge_l_p': rouge_l_p / dl_len,
#                     'test_rouge_l_r': rouge_l_r / dl_len}
#         wandb.log(log_dict)
#     md.train(True)
#     attacker.train(True)
#     return rouge_1 / dl_len, rouge_2 / dl_len, rouge_l_f1 / dl_len, rouge_l_p / dl_len, rouge_l_r / dl_len


def train_attacker(args):
    """
    训练攻击模型
    :param args:
    """
    config = FLConfig(collect_intermediates=False,
                      split_point_1=int(args.sps.split('-')[0]),
                      split_point_2=int(args.sps.split('-')[1]),
                      attack_mode=args.attack_mode,
                      noise_mode=args.noise_mode,
                      noise_scale_dxp=args.noise_scale_dxp,
                      noise_scale_gaussian=args.noise_scale_gaussian
                      )

    p = get_save_path(config, args.save_dir, args)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p) and args.skip_exists:
        print('Model exists, skipping...')
        return

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

    model.config_sfl(config,
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
    attack_model = LSTMDRAttacker(LSTMDRAttackerConfig(), model.config)
    if args.attack_model == 'lstm':
        attack_model = LSTMDRAttacker(LSTMDRAttackerConfig(), model.config)
    elif args.attack_model == 'gru':
        attack_model = GRUDRAttacker(LSTMDRAttackerConfig(), model.config)
    elif args.attack_model == 'linear':
        attack_model = LinearDRAttacker(DRAttackerConfig(), model.config)
    elif args.attack_model == 'dec':
        attack_model = DecoderDRAttacker(TransformerDRAttackerConfig(num_layers=args.md_n_layers), model.config)
    elif args.attack_model == 'attngru':
        attack_model = AttnGRUDRAttacker(TransformerGRUDRAttackerConfig(), model.config)
    elif args.attack_model == 'gruattn':
        attack_model = GRUAttnDRAttacker(TransformerGRUDRAttackerConfig(), model.config)
    elif args.attack_model == 'attn':
        attack_model = AttnDRAttacker(TransformerDRAttackerConfig(), model.config)
    if 'llama' not in args.model_name and 'chatglm' not in args.model_name:
        device = get_best_gpu()
        model.to(device)

    opt_cls = Adam
    if args.opt == 'adamw':
        opt_cls = AdamW
    optimizer = opt_cls(attack_model.parameters(), lr=args.lr, weight_decay=1e-5)
    attack_model.to(model.device)

    epoch = args.epochs
    if args.log_to_wandb:
        wandb.init(project=args.exp_name,
                   name=f"{args.model_name}_{args.split_point_1}",
                   config=vars(args)
                   )
    # evaluate(0, model, attack_model, tokenizer, dataloader_test, args.save_dir)
    with tqdm(total=epoch * len(dataloader)) as pbar:
        for epc in range(epoch):
            model.train(True)
            rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                if args.noise_mode == 'dxp' and args.noise_scale_dxp < 0:
                    # 随机生成噪声
                    model.change_noise(random_choose_noise(sfl.config.dxp_moe_range))
                elif args.noise_mode == 'gaussian' and args.noise_scale_gaussian < 0:
                    model.change_noise(random_choose_noise(sfl.config.gaussian_moe_range, mode='gaussian'))
                elif args.noise_mode == 'mix':
                    noise_mode = random.choice(['none', 'dxp', 'gaussian'])
                    if noise_mode == 'none':
                        model.change_noise(0, noise_mode)
                    elif noise_mode == 'dxp':
                        model.change_noise(random_choose_noise(dxp_moe_range, noise_mode), noise_mode)
                    else:
                        model.change_noise(random_choose_noise(gaussian_moe_range, mode='gaussian'), noise_mode)
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
                    # print(intermediate)
                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                loss.backward()
                optimizer.step()
                # 计算训练的ROGUE
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                rouge_1 += res['rouge-1']['f']
                rouge_2 += res['rouge-2']['f']
                rouge_l_f1 += res['rouge-l']['f']
                rouge_l_p += res['rouge-l']['p']
                rouge_l_r += res['rouge-l']['r']

                # print(logits.argmax(dim=-1))
                # print(tokenizer.decode(logits.argmax(dim=-1)[0], skip_special_tokens=False))

                pbar.set_description(
                    f'Epoch {epc} Loss {loss.item():.5f}, Rouge_Lf1 {rouge_l_f1 / (step + 1):.4f}')
                if step % 300 == 0 and model.type != 'encoder-decoder':
                    q = "To mix food coloring with sugar, you can"
                    print(q, "==>", get_output(q, model, attack_model))
                pbar.update(1)
                item_count += 1

            # 计算测试集上的ROGUE
            if (epc + 1) % args.checkpoint_freq == 0:
                evaluate(epc, model, attack_model, tokenizer, dataloader_test, args)
            if args.log_to_wandb:
                log_dict = {'epoch': epc,
                            'train_rouge_1': rouge_1 / item_count,
                            'train_rouge_2': rouge_2 / item_count,
                            'train_rouge_l_f1': rouge_l_f1 / item_count,
                            'train_rouge_l_p': rouge_l_p / item_count,
                            'train_rouge_l_r': rouge_l_r / item_count}
                wandb.log(log_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_dra_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    train_attacker(args)
