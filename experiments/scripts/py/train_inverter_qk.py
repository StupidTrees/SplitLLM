import argparse
import os
import sys

import numpy as np
import torch
import wandb
from torch.nn.functional import mse_loss
from torch.optim import Adam, AdamW
from tqdm import tqdm

from sfl.model.attacker.sip.inversion_models import InverterForAttentionConfig, LSTMDRAttackerConfig, \
    LSTMDRAttackerInterConfig, GRUInverterForAttention, GRUDRInverterWithAct, GRUDRInverter

sys.path.append(os.path.abspath('../../..'))
from experiments.scripts.py.eia_attack_qk import get_pi_size
from sfl.utils.args import FLConfig
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, required_quantization, \
    str2bool
from sfl.utils.model import get_best_gpu, calc_unshift_loss, set_random_seed, \
    evaluate_attacker_rouge

from sfl.utils.exp import get_dra_train_label, get_dra_test_label


def get_attacker_path(args, fl_config, save_dir):
    model_name = args.model_name
    cut_layer = fl_config.split_point_1 if fl_config.attack_mode == "b2tr" else fl_config.split_point_2
    p = os.path.join(save_dir,
                     f'{model_name}/{args.train_dataset}/{get_dra_train_label(args.train_dataset)}*{args.dataset_train_frac:.3f}-{get_dra_test_label(args.train_dataset)}*{args.dataset_test_frac:.3f}'
                     f'/{args.attack_model}/layer{cut_layer}/')
    attacker_prefix = f'pi-{args.target}/'
    p += attacker_prefix
    return p


def get_attacker(args, fl_config: FLConfig, save_dir):
    p = get_attacker_path(args, fl_config, save_dir)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p):
        print('Model exists, returning...')
        l = sorted(list(os.listdir(p)), key=lambda x: float(x.split('_')[-1]))[-1]
        attacker_path_1 = os.path.join(p, l)
        assert os.path.exists(attacker_path_1)
        return get_attacker_class(args).from_pretrained(attacker_path_1)
    return None


def train_attacker(model, tokenizer, args):
    model.train(False)
    dataset = get_dataset_class(args.train_dataset)(tokenizer, [], uni_length=args.uni_length)
    uni_length = args.uni_length
    if get_dra_train_label(args.train_dataset) == get_dra_test_label(args.train_dataset):  # self-testing
        dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                                      type=get_dra_train_label(args.train_dataset),
                                                                      shrink_frac=args.dataset_train_frac,
                                                                      further_test_split=0.3)
    else:
        dataloader = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                     type=get_dra_train_label(args.train_dataset),
                                                     shrink_frac=args.dataset_train_frac)
        dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                          type=get_dra_test_label(args.train_dataset),
                                                          shrink_frac=args.dataset_test_frac)

    # # TODO-REMOVE
    # dataloader_test = get_dataset_class(args.dataset)(tokenizer, [],
    #                                                   uni_length=args.uni_length).get_dataloader_unsliced(batch_size=4,
    #                                                                                                       type='train',
    #                                                                                                       shrink_frac=1.0)
    # 开始训练Attack Model
    # Generate Pi
    atk_clz = get_attacker_class(args)
    if args.target == 'qk':
        attack_model = atk_clz(InverterForAttentionConfig(hidden_size=256, bidirectional=False, uni_length=uni_length),
                               target_config=model.config)
    else:
        cfg = LSTMDRAttackerConfig(hidden_size=256, bidirectional=False)
        if args.target == 'o5':
            cfg = LSTMDRAttackerInterConfig(hidden_size=256, bidirectional=False)
        cfg.vocab_size = model.config.vocab_size
        name_or_path = model.config.name_or_path
        if '/' in name_or_path:
            if name_or_path.endswith('/'):
                name_or_path = name_or_path[:-1]
            name_or_path = name_or_path.split('/')[-1]
        cfg.target_model = name_or_path
        if args.target == 'o5':
            cfg.inter_size = get_pi_size(args, model)
            attack_model = atk_clz(cfg, target_config=model.config)
        else:
            cfg.n_embed = get_pi_size(args, model)
            attack_model = atk_clz(cfg, target_config=None)


    # if args.target == 'o5':
    #     lr = 2e-3
    #     wd = 1e-3

    attack_model.to(model.device)
    if args.target == 'o5':
        o5_epcs = 3
        for name, param in attack_model.named_parameters():
            if 'red' not in name:
                param.requires_grad = False
        optimizer = Adam([p for p in attack_model.parameters() if p.requires_grad], lr=0.001, weight_decay=1e-4)
        with tqdm(total=o5_epcs * len(dataloader)) as pbar:
            for epc in range(o5_epcs):
                model.train(False)
                for step, batch in enumerate(dataloader):
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(model.device)
                    attention_mask = batch['attention_mask'].to(model.device)
                    hidden_state, intermediate = model(input_ids=input_ids, attention_mask=attention_mask)
                    # print(intermediate)
                    pred_o6, logits = attack_model(intermediate['o5'])
                    loss = mse_loss(pred_o6, intermediate['o6'])
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(
                        f'O5-EP {epc} Loss {loss.item():.3f}')
                    pbar.update(1)
        for name,param in attack_model.named_parameters():
            param.requires_grad = True
    opt_cls = Adam
    if args.opt == 'adamw':
        opt_cls = AdamW
    lr = args.lr
    wd = 1e-5
    optimizer = opt_cls(attack_model.parameters(), lr=lr, weight_decay=wd)
    with tqdm(total=args.epochs * len(dataloader)) as pbar:
        for epc in range(args.epochs):
            model.train(False)
            rglf = 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                hidden_state, intermediate = model(input_ids=input_ids, attention_mask=attention_mask)
                # print(intermediate)
                attn = merge_attention(args, (hidden_state, intermediate))
                atk_outputs = attack_model(attn)
                if args.target == 'qk':
                    pred_hidden, logits = atk_outputs
                    loss = calc_unshift_loss(logits, input_ids)
                elif args.target == 'o5':
                    pred_o6, logits = atk_outputs
                    loss = calc_unshift_loss(logits, input_ids)
                    loss += mse_loss(pred_o6, intermediate['o6'])
                else:
                    logits = atk_outputs
                    loss = calc_unshift_loss(logits, input_ids)
                loss.backward()
                optimizer.step()
                # 计算训练的ROGUE
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                rglf += res['rouge-l']['f']
                pbar.set_description(
                    f'Epoch {epc} Loss {loss.item():.3f}, Rouge_Lf1 {rglf / (step + 1):.4f}')
                pbar.update(1)
                item_count += 1
            if (epc + 1) % args.checkpoint_freq == 0:
                evaluate(model, attack_model, tokenizer, dataloader_test, None, args, pi=None, epc=epc,
                         mode="validation", max_samples=20)
            if args.log_to_wandb and not args.debug:
                log_dict = {'epoch': epc,
                            'train_rouge_l_f1': rglf / item_count}
                wandb.log(log_dict)


def evaluate(md, attacker, tok, test_data_loader, sample_batch, args, pi=None, mode='validation', epc=0,
             max_samples=1000):
    """
    恢复的评价指标选用ROUGE
    :return: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-P, ROUGE-L-R
    """
    md.eval()
    attacker.eval()
    dl_len = 0
    with torch.no_grad():
        rouge_l_f = 0
        met = 0
        tok_acc = 0
        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            input_ids = batch['input_ids'].to(md.device)
            attention_mask = batch['attention_mask'].to(md.device)
            outputs = md(input_ids=input_ids, attention_mask=attention_mask)
            atk_outpus = attacker(merge_attention(args, outputs, pi=pi))
            if args.target in ['qk','o5']:
                _, logits = atk_outpus
            else:
                logits = atk_outpus
            result, m, t = evaluate_attacker_rouge(tok, logits, batch)
            rouge_l_f += result['rouge-l']['f']
            met += m
            tok_acc += t
            dl_len += 1
            if step > max_samples:
                break

    print(
        f'Epoch {epc} Test Rouge_l_f1: {rouge_l_f / dl_len}')  # , Test2 Rouge_l_f1: {rouge_l_f1_x / dl_len if attacker2 else 0}')
    if mode == 'validation':
        p = get_attacker_path(args, md.fl_config, args.save_dir)
        if args.save_checkpoint:
            attacker.save_pretrained(p + f'epoch_{epc}_rouge_{rouge_l_f / dl_len:.4f}')

    if args.log_to_wandb and not args.debug:
        log_dict = {f'{mode}_epoch': epc,
                    f'{mode}_rouge_l_f1': rouge_l_f / dl_len,
                    f'{mode}_meteor': met / dl_len, f'{mode}_tokacc': tok_acc / dl_len,
                    }
        if sample_batch is not None:
            outputs = md(sample_batch['input_ids'].to(md.device))
            atk_outputs = attacker(merge_attention(args, outputs, pi=pi).to(md.device))
            if args.target in ['qk','o5']:
                attacked = atk_outputs[1]
            else:
                attacked = atk_outputs
            texts = [tok.decode(t.argmax(-1), skip_special_tokens=False) for t in attacked]
            if args.debug:
                print('REC:', texts[0])
                print('GT :', sample_batch['input_text'][0])
            table = wandb.Table(
                columns=["attacked_text", "true_text"],
                data=[[txt, gt] for txt, gt in zip(texts, sample_batch['input_text'])])
            log_dict['attack_sample'] = table
        wandb.log(log_dict)

    attacker.train(True)
    return rouge_l_f / dl_len


def merge_attention(args, outputs, pi=None):
    hidden_states, other_output = outputs
    targets = ['qk', 'o4', 'o5', 'o6']
    if args.target in targets:
        target = other_output[args.target]
    else:
        target = hidden_states

    if args.mode == 'perm' and pi is not None:
        if pi == 'g':
            p = np.random.permutation(target.size(-1))
            pi = torch.eye(target.size(-1), target.size(-1))[:, p].cuda()
        target = target @ pi.to(target.device)

    elif args.mode == 'random' and pi is not None:
        target = torch.rand_like(target).to(target.device)

    return target


def get_attacker_class(args):
    if args.target == 'qk':
        # if args.uni_length > 0:
        #     return GRUInverterForAttentionUni
        # else:
        return GRUInverterForAttention
    elif args.target == 'o5':
        return GRUDRInverterWithAct
    else:
        return GRUDRInverter


def main(args):
    """
    训练攻击模型
    :param args:
    """
    config = FLConfig(collect_intermediates=False,
                      split_point_1=int(args.sps.split('-')[0]),
                      split_point_2=int(args.sps.split('-')[1]),
                      attack_mode='b2tr',
                      noise_mode=args.noise_mode,
                      noise_scale=args.noise_scale,
                      split_mode='attention'
                      )

    model, tokenizer = get_model_and_tokenizer(args.model_name, load_bits=args.load_bits)

    model.config_sfl(config)
    # freeze all parts:
    for name, param in model.named_parameters():
        param.requires_grad = False
    if not required_quantization(args.model_name) and model.device == 'cpu':
        model.to(get_best_gpu())
    model.train(False)

    if args.log_to_wandb and not args.debug:
        wandb.init(project='>[PI]ATK_QK.sh',
                   name=f"SD[{args.seed}]{args.model_name}_{args.sps}_{args.dataset}<<{args.target}-{args.mode}",
                   config=vars(args)
                   )

    dataset = get_dataset_class(args.dataset)(tokenizer, [], uni_length=args.uni_length)
    sample_batch = next(iter(dataset.get_dataloader_unsliced(6, 'train',shuffle=False)))
    attacker = get_attacker(args, config, args.save_dir)
    if attacker is None:
        train_attacker(model, tokenizer, args)
        attacker = get_attacker(args, config, args.save_dir)
    assert attacker is not None
    attacker.to(model.device)
    n_embed = get_pi_size(args, model)
    if args.debug:
        # pi = torch.ran
        avg = 0
        for i in range(10):
            # p = np.random.permutation(n_embed)
            # pi = torch.eye(n_embed, n_embed)[:, p].cuda()
            # pi = torch.rand_like(pi)

            # make 30% of pi's random position
            # for i in range(int(n_embed * 0.5)):
            #     pi[i, i] = 1
            # make 30% random element of pi 1
            pi = torch.zeros(n_embed, n_embed)
            indexes = torch.randint(0, n_embed, (int(n_embed * 0.003),))
            for i in indexes:
                pi[i, i] = 1
            dataloader_test = dataset.get_dataloader_unsliced(batch_size=4,
                                                              type='train',
                                                              shrink_frac=1.0)
            rgl = evaluate(model, attacker, tokenizer, dataloader_test, sample_batch, args, mode="test", pi=pi,
                           max_samples=10)
            avg += rgl
        print('AVG:', avg / 10)
    else:
        if args.uni_length > 0:
            p = np.random.permutation(n_embed)
            pi = torch.eye(n_embed, n_embed)[:, p].cuda()
        else:
            pi = 'g'

        dataloader_test = dataset.get_dataloader_unsliced(batch_size=4,
                                                          type='train',
                                                          shrink_frac=1.0)
        evaluate(model, attacker, tokenizer, dataloader_test, sample_batch, args, mode="test", pi=pi,
                 max_samples=args.samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_dra_params(parser)
    parser.add_argument('--target', type=str, default='qk', help='target of attack')
    parser.add_argument('--mode', type=str, default='random', help='mode of attack')
    parser.add_argument('--train_dataset', type=str, default='sensireplaced')
    parser.add_argument('--uni_length', type=int, default=-1)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--debug', type=str2bool, default=False)
    args = parser.parse_known_args()[0]
    set_random_seed(args.seed)
    main(args)
