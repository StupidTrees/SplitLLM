import argparse
import copy
import os
import pickle
import random
import sys

import torch
import wandb
from peft import PeftModel
from torch.optim import Adam
from tqdm import tqdm

sys.path.append(os.path.abspath('../../..'))
from sfl.simulator.param_keeper import InMemoryParameterKeeper
from sfl.data.base import MixtureFedDataset
from experiments.scripts.py.train_inverter import get_save_path, evaluate, pre_no_peek, get_lora_path
from sfl.config import FLConfig, DRA_test_label, DRA_train_label, dxp_moe_range, gaussian_moe_range, dc_moe_range
from sfl.model.attacker.sip_attacker import MOEDRInverter, MOEDRAttackerConfig
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, add_train_dra_params
from sfl.utils.model import get_t5_input, get_best_gpu, calc_unshift_loss, set_random_seed, \
    evaluate_attacker_rouge, random_choose_noise, ParamRestored


def calc_pk_norm(pk: InMemoryParameterKeeper, key=''):
    norm = 0
    for part in ['bottom', 'trunk', 'top']:
        for p in pk.get_other_params(key, part):
            norm += p.norm().item()
    return norm


def calc_model_norm(model):
    norm = 0
    for p in model.parameters():
        if p.requires_grad:
            norm += p.norm().item()
    return norm


def llm_forward(args, model, batch, tokenizer):
    if model.type == 'encoder-decoder':
        input_args = get_t5_input(batch, tokenizer, model.device)
        enc_hidden, dec_hidden = model(**input_args)
        intermediate = torch.concat([enc_hidden, dec_hidden], dim=1)
        input_ids = torch.concat([input_args['input_ids'], input_args['decoder_input_ids']],
                                 dim=1).to(model.device)
    else:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        intermediate = model(input_ids=input_ids, attention_mask=attention_mask)
    return input_ids, intermediate


def experts_no_peek_pre(args, expert_scales, llm, dataloader, random_num=18):
    save_dir = get_lora_path(args, 'dc')
    result = {}
    parameter_keeper = InMemoryParameterKeeper([])
    if not isinstance(llm, PeftModel):
        llm = llm.convert_to_lora_model(restore_rest=False)
    for key in ['pretrained']:
        parameter_keeper.store_other_params(key, 'trunk',
                                            [p.detach().cpu() for nm, p in llm.get_trunk_params()])
        parameter_keeper.store_other_params(key, 'top',
                                            [p.detach().cpu() for nm, p in llm.get_top_params()])
        parameter_keeper.store_other_params(key, 'bottom',
                                            [p.detach().cpu() for nm, p in llm.get_bottom_params()])
    # print('PK_NORM_INIT:', calc_pk_norm(parameter_keeper, 'pretrained'))
    # print('MODEL_NORM_INIT:', calc_model_norm(llm))

    # read all files under save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fn in os.listdir(save_dir):
        scale = round(float('.'.join(fn.split('_')[-1].split('.')[:-1])), 2)
        pk = pickle.load(open(os.path.join(save_dir, fn), 'rb'))
        # pk = copy.deepcopy(parameter_keeper)
        # with ParamRestored(llm=llm, param_keeper=pk, parts=['bottom', 'trunk', 'top'], write_back=True,
        #                    disable_inter_collection=True):
        #     llm.load_adapter(os.path.join(save_dir, fn), 'lora', True)
        #     llm.set_adapter('lora')
        #     print(f'MODEL_NORM_{scale}:', calc_model_norm(llm))
        print(f'PK_NORM_{scale}:', calc_pk_norm(pk))
        result[scale] = pk
    if len(result) >= len([x for x in expert_scales if x >= 0]) + random_num:
        left_scales = set(result.keys()) - set(expert_scales)
        return result, list(left_scales)
    random_scales = []
    left_random_num = set(result.keys()) - set([x for x in expert_scales if x >= 0])
    random_num -= left_random_num
    for i in range(random_num):
        random_scales.append(round(random.uniform(min(dc_moe_range) / 2, max(dc_moe_range) * 1.5), 2))

    for scale in expert_scales + random_scales:
        if scale < 0 or scale in result:
            print(f'Skipping {scale}')
            continue
        pk = copy.deepcopy(parameter_keeper)
        with ParamRestored(llm=llm, param_keeper=pk, parts=['bottom', 'trunk', 'top'], write_back=True,
                           disable_inter_collection=True):
            pre_no_peek(scale, llm, dataloader)
        pickle.dump(pk, open(os.path.join(save_dir, f'nopeek_{scale:.2f}.pkl'), 'wb'))
        # llm.save_pretrained(os.path.join(save_dir, f'nopeek_{scale:.2f}'))
        # print(f'MODEL_NORM_{scale}:', calc_model_norm(llm))
        # print(f'PK1_NORM_{scale}:', calc_pk_norm(pk))
        # print(f'PK2_NORM_{scale}:', calc_pk_norm(pk))
        result[scale] = pk

    return result, random_scales


def train_attacker(args):
    """
    训练攻击模型
    :param args:
    """
    model, tokenizer = get_model_and_tokenizer(args.model_name, load_bits=args.load_bits)

    def get_output(text, encoder_model, attack_model):
        t = tokenizer(text, return_tensors="pt")
        inter = encoder_model(t['input_ids'].to(model.device), attention_mask=t['attention_mask'].to(model.device))
        res = attack_model(inter)
        r = tokenizer.decode(res.argmax(dim=-1)[-1], skip_special_tokens=True)
        return r

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
    #
    #                                                                                        shrink_frac=0.01)
    config = FLConfig(collect_intermediates=False,
                      split_point_1=int(args.sps.split('-')[0]),
                      split_point_2=int(args.sps.split('-')[1]),
                      attack_mode=args.attack_mode,
                      noise_mode=args.noise_mode,
                      use_lora_at_top=True,
                      use_lora_at_trunk=True,
                      use_lora_at_bottom=True
                      )
    model.config_sfl(config,
                     param_keeper=None)
    model = model.convert_to_lora_model(restore_rest=False)

    expert_scales = [0, -1] + list(dxp_moe_range)
    if args.noise_mode == 'gaussian':
        expert_scales = [0, -1] + list(gaussian_moe_range)
    elif args.noise_mode == 'dc':
        expert_scales = [0, -1] + list(dc_moe_range)
    expert_modes = [args.noise_mode] * len(expert_scales)
    if args.attack_model == 'moe2':
        # 三个专家，一个混dxp，一个混GAUSSIAN
        expert_modes = ['none', 'dxp', 'gaussian']
        expert_scales = [0, tuple(dxp_moe_range), tuple(gaussian_moe_range)]

    p = get_save_path(config, args.save_dir, args)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p) and args.skip_exists:
        print('Model exists, skipping...')
        return
    extra_noise_scales = None
    nopeek_pks = None
    if args.noise_mode == 'dc':
        nopeek_pks, random_scales = experts_no_peek_pre(args, expert_scales, model, dataloader)
        for scale, pk in nopeek_pks.items():
            print(f'NOPEEK {scale} {calc_pk_norm(pk)}')

        extra_noise_scales = random_scales

    # 开始训练Attack Model

    attack_model = MOEDRInverter(MOEDRAttackerConfig(expert_scales=expert_scales, hidden_size=256), model.config)

    if 'llama' not in args.model_name:
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
    with tqdm(total=args.epochs * len(dataloader)) as pbar:
        for epc in range(args.epochs):
            model.train(True)
            rougeLs = [0] * len(attack_model.config.expert_scales)
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                if args.attack_model == 'moe':
                    inters = []
                    _, std_inter = llm_forward(args, model, batch, tokenizer)
                    for noise_scale in attack_model.config.expert_scales:
                        if noise_scale < 0:
                            noise_scale = random_choose_noise(expert_scales, mode=args.noise_mode,
                                                              extra_choices=extra_noise_scales)
                        if args.noise_mode == 'dc':
                            with ParamRestored(llm=model, param_keeper=nopeek_pks[noise_scale],
                                               parts=['bottom', 'trunk', 'top'],
                                               write_back=False, disable_inter_collection=False):
                                input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                        else:
                            model.change_noise(noise_scale)
                            input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                        inters.append(intermediate)
                elif args.attack_model == 'moe2':
                    inters = []
                    for noise_mode, scales in zip(expert_modes, expert_scales):
                        if noise_mode == 'none':
                            noise_scale = 0
                        else:
                            noise_scale = random_choose_noise(scales, mode=noise_mode, extra_choices=extra_noise_scales)
                        if noise_mode != 'dc':
                            model.change_noise(noise_scale, mode=noise_mode)
                            input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                        else:
                            with ParamRestored(llm=model, param_keeper=nopeek_pks[noise_scale],
                                               parts=['bottom', 'trunk', 'top'],
                                               write_back=False, disable_inter_collection=False):
                                input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                        inters.append(intermediate)
                # rg_rpt = ", ".join([f"Exp{i}_MSE{mse_loss(inter, std_inter).item()}" for i, inter in enumerate(inters)])
                # print(rg_rpt)
                exp_logits = attack_model.train_exp_forward(inters)
                loss = 0
                for i, logits in enumerate(exp_logits):
                    if logits is None:
                        continue
                    loss += calc_unshift_loss(logits, input_ids)
                    res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                    rougeLs[i] += res['rouge-l']['f']
                loss.backward()
                optimizer.step()

                # 计算训练的ROGUE
                rg_rpt = ", ".join(
                    [f"Exp{i}-[{expert_scales[i]}]-{rgl / (step + 1):.3f}" for i, rgl in enumerate(rougeLs)])
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
                    random_noise = random_choose_noise(attack_model.config.expert_scales, mode=args.noise_mode,
                                                       extra_choices=extra_noise_scales)
                    if args.noise_mode != 'dc':
                        model.change_noise(random_noise)
                        input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                    else:
                        with ParamRestored(llm=model, param_keeper=nopeek_pks[random_noise],
                                           parts=['bottom', 'trunk'],
                                           write_back=False, disable_inter_collection=False):
                            input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                elif args.attack_model == 'moe2':
                    noise_mode = random.choice(expert_modes)
                    if noise_mode == 'none':
                        noise_scale = 0
                    elif noise_mode == 'dxp':
                        noise_scale = random_choose_noise(expert_scales[1], mode=noise_mode)
                    elif noise_mode == 'gaussian':
                        noise_scale = random_choose_noise(expert_scales[2], mode=noise_mode)
                    model.change_noise(noise_scale, mode=noise_mode)
                    input_ids, intermediate = llm_forward(args, model, batch, tokenizer)

                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
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
                evaluate(epc, model, attack_model, tokenizer, dataloader_test, args)
            if args.log_to_wandb:
                log_dict = {'epoch': epc, 'Total_RgL': rougeL_total / item_count}
                wandb.log(log_dict)

    # 阶段3，训练整体模型
    attack_model.freeze_parts(experts=True, freeze=False)  # freeze experts
    attack_model.freeze_parts(experts=False, freeze=False)  # freeze gating
    optimizer3 = Adam([p for p in attack_model.parameters() if p.requires_grad], lr=args.lr)
    epochs_ft = args.epochs_ft
    with tqdm(total=epochs_ft * len(dataloader)) as pbar:
        for epc in range(epochs_ft):
            model.train(True)
            rougeL_total = 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer2.zero_grad()
                if args.attack_model == 'moe':
                    # 随机噪声规模
                    random_noise = random_choose_noise(attack_model.config.expert_scales, mode=args.noise_mode,
                                                       extra_choices=extra_noise_scales)
                    if args.noise_mode != 'dc':
                        model.change_noise(random_noise)
                        input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                    else:
                        with ParamRestored(llm=model, param_keeper=nopeek_pks[random_noise],
                                           parts=['bottom', 'trunk'],
                                           write_back=False, disable_inter_collection=False):
                            input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                elif args.attack_model == 'moe2':
                    noise_mode = random.choice(expert_modes)
                    if noise_mode == 'none':
                        noise_scale = 0
                    elif noise_mode == 'dxp':
                        noise_scale = random_choose_noise(expert_scales[1], mode=noise_mode)
                    elif noise_mode == 'gaussian':
                        noise_scale = random_choose_noise(expert_scales[2], mode=noise_mode)
                    model.change_noise(noise_scale, mode=noise_mode)
                    input_ids, intermediate = llm_forward(args, model, batch, tokenizer)
                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                rougeL_total += res['rouge-l']['f']
                loss.backward()
                optimizer3.step()

                # 计算训练的ROGUE
                pbar.set_description(
                    f'FINAL Epoch {epc} Loss {loss.item():.5f}, RougeL {rougeL_total / (step + 1):.4f}')
                if step % 300 == 0 and model.type != 'encoder-decoder':
                    q = "To mix food coloring with sugar, you can"
                    print(q, "==>", get_output(q, model, attack_model))
                pbar.update(1)
                item_count += 1
            # 计算测试集上的ROGUE
            if (epc + 1) % 1 == 0:
                evaluate(epc * 100, model, attack_model, tokenizer, dataloader_test, args)
            if args.log_to_wandb:
                log_dict = {'epoch': epc, 'Total_RgL': rougeL_total / item_count}
                wandb.log(log_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_dra_params(parser)
    parser.add_argument('--epochs_ft', type=int, default=10)
    args = parser.parse_args()
    set_random_seed(args.seed)
    train_attacker(args)
