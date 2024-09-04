import argparse
import os
import sys

import numpy as np
import torch
import wandb
from tqdm import tqdm

sys.path.append(os.path.abspath('../../..'))
from sfl.config import FLConfig
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, add_train_dra_params, required_quantization
from sfl.utils.model import get_best_gpu, set_random_seed, \
    evaluate_attacker_rouge, get_embedding_layer, get_embed_size, get_embedding_matrix

from sfl.config import DRA_train_label, DRA_test_label


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


def rec_text(llm, embeds):
    wte = get_embedding_layer(llm)
    all_words = torch.tensor(list([i for i in range(llm.config.vocab_size)])).to(llm.device)
    all_embeds = wte(all_words)
    if all_embeds.dtype == torch.float16:
        embeds = embeds.float()
        all_embeds = all_embeds.float()
    cosine_similarities = torch.matmul(embeds, all_embeds.transpose(0, 1))  # (bs, seq,vocab)
    return torch.softmax(cosine_similarities, -1)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits#  + sample_gumbel(logits.size()).to(logits.device)
    return torch.softmax(y / temperature, dim=-1)


def get_pi_size(args, model):
    if args.target == 'qk':
        n_embed = args.uni_length
    else:
        if args.target in ['o4', 'o6']:
            n_embed = get_embed_size(model.config)
        else:
            if hasattr(model.config, 'intermediate_size'):
                n_embed = model.config.intermediate_size
            elif hasattr(model.config, 'n_inner'):
                if model.config.n_inner is not None:
                    n_embed = model.config.n_inner
                else:
                    n_embed = 4 * get_embed_size(model.config)
    return n_embed


def eia(args, model, tokenizer, pi, batch):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    attn = merge_attention(args, model_outputs, pi=pi).detach().clone()
    # bs, n_head, seq_len, seq_len = intermediate[0].shape
    # attn = intermediate[-1].contiguous().view(bs, seq_len, -1).detach().clone()
    pbar = tqdm(total=args.epochs)
    avg_rglf = 0
    # dummy = torch.randn_like(hidden_state)
    embedding_layer = get_embedding_layer(model)
    embedding_matrix = get_embedding_matrix(model)
    if args.method == 'eia':
        bs, sl = model_outputs[0].shape[:-1]
        dummy = torch.rand((bs, sl, model.config.vocab_size)).to(model.device)
    else:
        dummy = torch.randint(0, model.config.vocab_size, model_outputs[0].shape[:-1]).to(model.device)
        dummy = dummy.long()
        dummy = embedding_layer(dummy)
        # dummy = torch.rand_like(dummy).to(model.device)
    dummy.requires_grad = True
    opt = torch.optim.AdamW([dummy], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=args.wd)
    for e in range(args.epochs):
        opt.zero_grad()
        dmy = dummy
        if args.method == 'eia':
            dmy = gumbel_softmax_sample(dummy.float(), args.temp) @ embedding_matrix
        outputs = model(inputs_embeds=dmy)
        it = merge_attention(args, outputs)
        # bs, n_head, seq_len, seq_len = it[0].shape
        # it = it[-1].contiguous().view(bs, seq_len, -1)
        target = attn.to(model.device)
        loss = 0
        for x, y in zip(it, target):
            if args.target == 'o6':
                loss += 1 - torch.cosine_similarity(x, y, dim=-1).sum()
            else:
                loss += ((x - y) ** 2).sum()
        loss.backward()
        opt.step()
        if e % 200 == 0 and e != 0:
            if args.method == 'eia':
                texts = dummy
            else:
                texts = rec_text(model, dummy)
            rg, _, _ = evaluate_attacker_rouge(tokenizer, texts, batch)
            avg_rglf = rg["rouge-l"]["f"]
        pbar.set_description(
            f'Epoch {e}/{args.epochs} Loss: {loss.item()} ROUGE: {avg_rglf}')
        pbar.update(1)
    if args.method == 'eia':
        texts = dummy
    else:
        texts = rec_text(model, dummy)
    return texts


def train_attacker(args):
    """
    训练攻击模型
    :param args:
    """
    config = FLConfig(collect_intermediates=False,
                      split_point_1=int(args.sps.split('-')[0]),
                      split_point_2=int(args.sps.split('-')[1]),
                      attack_mode='b2tr',
                      noise_mode=args.noise_mode,
                      noise_scale_dxp=args.noise_scale_dxp,
                      noise_scale_gaussian=args.noise_scale_gaussian,
                      split_mode='attention'
                      )

    model, tokenizer = get_model_and_tokenizer(args.model_name, load_bits=args.load_bits)

    dataset_cls = get_dataset_class(args.dataset)
    dataset = dataset_cls(tokenizer, [], uni_length=args.uni_length)
    if DRA_train_label[args.dataset] == DRA_test_label[args.dataset]:  # self-testing
        _, dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                             type=DRA_train_label[args.dataset],
                                                             shrink_frac=args.dataset_train_frac,
                                                             further_test_split=0.3)
    else:
        # dataloader = dataset.get_dataloader_unsliced(batch_size=args.batch_size, type=DRA_train_label[args.dataset],
        #                                              shrink_frac=args.dataset_train_frac)
        dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                          type=DRA_test_label[args.dataset],
                                                          shrink_frac=args.dataset_test_frac)

    model.config_sfl(config)
    # freeze all parts:
    for name, param in model.named_parameters():
        param.requires_grad = False

    if not required_quantization(args.model_name) and model.device == 'cpu':
        model.to(get_best_gpu())

    if args.log_to_wandb:
        wandb.init(project='>[PI]EIA_QK.sh',
                   name=f"==SD[{args.seed}][{args.method}]-{args.model_name}_{args.dataset}_{args.sps}<[{args.target}]-[{args.mode}]",
                   config=vars(args)
                   )
    # Generate Pi
    n_dim = get_pi_size(args,model)
    p = np.random.permutation(n_dim)
    pi = torch.eye(n_dim, n_dim)[:, p].cuda()
    # evaluate(0, model, attack_model, tokenizer, dataloader_test, args.save_dir)
    sample_batch = next(iter(dataset.get_dataloader_unsliced(6, 'train',shuffle=False)))
    sample_texts = eia(args,model,tokenizer,pi,sample_batch)
    texts = [tokenizer.decode(t.argmax(-1), skip_special_tokens=True) for t in sample_texts]
    table = wandb.Table(
        columns=["attacked_text", "true_text"],
        data=[[txt, gt] for txt, gt in zip(texts, sample_batch['input_text'])])
    wandb.log({'sample_text': table})
    with tqdm(total=args.sample_num):
        model.train(False)
        item_count = 0
        avg_rgL = 0
        avg_met = 0
        avg_tok = 0
        for step, batch in enumerate(dataloader_test):
            texts = eia(args,model,tokenizer,pi,batch)
            rg, meteor, tok_acc = evaluate_attacker_rouge(tokenizer, texts, batch)
            avg_rgL += rg["rouge-l"]["f"]
            avg_met += meteor
            avg_tok += tok_acc
            item_count += 1
            if args.log_to_wandb:
                log_dict = {
                    'avg_rgLf': avg_rgL / item_count,
                    'avg_METEOR': avg_met / item_count,
                    'avg_TOK_ACC': avg_tok / item_count
                }
                wandb.log(log_dict)
            if (step + 1) > args.sample_num:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_dra_params(parser)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--target', type=str, default='qk', help='target of attack')
    parser.add_argument('--method', type=str, default='bre', help='target of attack')
    parser.add_argument('--temp', type=float, default=1.2)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--mode', type=str, default='random', help='mode of attack')
    parser.add_argument('--train_dataset', type=str, default='sensireplaced')
    parser.add_argument('--uni_length', type=int, default=-1)
    args = parser.parse_known_args()[0]
    set_random_seed(args.seed)
    train_attacker(args)
