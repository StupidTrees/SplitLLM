import argparse
import copy
import os
import pickle
import random
import sys

import torch
import wandb
from peft import PeftModel
from torch.optim import Adam, AdamW
from tqdm import tqdm

from sfl.model.attacker.sip.args import SIPAttackerArguments, InversionModelTrainingArgument
from sfl.model.attacker.sip.inversion_models import get_inverter_with_config, MOEDRInverter, MOEDRAttackerConfig
from sfl.model.reducer.dim_reducer import get_dim_reducer
from sfl.simulator.param_keeper import InMemoryParameterKeeper
from sfl.utils.exp import required_quantization, get_dataset_class

sys.path.append(os.path.abspath('../../..'))
import sfl
from sfl.data.base import MixtureFedDataset
from sfl.utils.args import FLConfig, dxp_moe_range, gaussian_moe_range, lora_path, dc_moe_range
from sfl.utils.model import get_t5_input, calc_unshift_loss, evaluate_attacker_rouge, random_choose_noise, \
    FLConfigHolder, dist_corr, ParamRestored

from sfl.utils.exp import get_dra_train_label, get_dra_test_label


def get_lora_path(sip_arg: SIPAttackerArguments, type):
    model_name = sip_arg.target_model_name
    if required_quantization(model_name):
        model_name += f'-{sip_arg.target_model_load_bits}bits'
    p = os.path.join(lora_path,
                     f'{model_name}/{sip_arg.dataset}/{type}/')
    return p


def get_save_path(fl_config, save_dir, args):
    model_name = args.model_name
    cut_layer = fl_config.split_point_1 if fl_config.attack_mode == "b2tr" else fl_config.split_point_2
    if required_quantization(args.model_name):
        model_name += f'-{args.load_bits}bits'
    if ',' in args.dataset:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/Tr{args.dataset_train_frac:.3f}-Ts*{args.dataset_test_frac:.3f}'
                         f'/{args.attack_model}/layer{cut_layer}/')
    else:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/{get_dra_train_label(args.dataset)}*{args.dataset_train_frac:.3f}-{get_dra_test_label(args.dataset)}*{args.dataset_test_frac:.3f}'
                         f'/{args.attack_model}/layer{cut_layer}/')
    attacker_prefix = 'normal/'
    if fl_config.noise_mode == 'dxp':
        attacker_prefix = f'{fl_config.noise_mode}:{fl_config.noise_scale}/'
    elif fl_config.noise_mode == 'gaussian':
        attacker_prefix = f'{fl_config.noise_mode}:{fl_config.noise_scale}/'
    elif fl_config.noise_mode == 'dc':
        attacker_prefix = f'{fl_config.noise_mode}:{fl_config.noise_scale}/'
    elif fl_config.noise_mode == 'mix':
        attacker_prefix = 'mix/'
    if 'moe' in args.attack_model:
        attacker_prefix = f'{fl_config.noise_mode}/'
    if args.require_prefix is not None:
        attacker_prefix = f'{args.require_prefix}/'
    p += attacker_prefix
    return p


def get_inverter_path(sip_args: SIPAttackerArguments, training_args: InversionModelTrainingArgument, b2tr=True):
    layer = sip_args.b2tr_layer if b2tr else sip_args.tr2t_layer
    model_name = sip_args.target_model_name
    if required_quantization(sip_args.target_model_name):
        model_name += f'-{sip_args.target_model_load_bits}bits'
    if ',' in sip_args.dataset:
        p = os.path.join(sip_args.path,
                         f'{model_name}/{sip_args.dataset}/Tr{sip_args.train_frac:.3f}-Ts*{training_args.test_frac:.3f}'
                         f'/{sip_args.model}/layer{layer}/')
    else:
        p = os.path.join(sip_args.path,
                         f'{model_name}/{sip_args.dataset}/{get_dra_train_label(sip_args.dataset)}*{sip_args.train_frac:.3f}-{get_dra_test_label(sip_args.dataset)}*{training_args.test_frac:.3f}'
                         f'/{sip_args.model}/layer{layer}/')
    p += sip_args.prefix + '/'
    return p


def evaluate(sip_args: SIPAttackerArguments, training_args: InversionModelTrainingArgument, b2tr,
             epc, md, attacker, tok, test_data_loader):
    md.eval()
    noise_mode, noise_scale = _resolve_noise_args(sip_args.prefix)
    if noise_mode == 'mix' or noise_scale < 0:
        md.change_noise(0)
    attacker.eval()
    dl_len = len(test_data_loader)
    with torch.no_grad():
        rouge_l_f = 0
        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            if md.type == 'encoder-decoder':
                enc_hidden, dec_hidden = md(**get_t5_input(batch, tok, md.device))
                inter = torch.concat([enc_hidden, dec_hidden], dim=1)
            else:
                input_ids = batch['input_ids'].to(md.device)
                attention_mask = batch['attention_mask'].to(md.device)
                inter = md(input_ids=input_ids, attention_mask=attention_mask)
            logits = attacker(inter)
            result, _, _ = evaluate_attacker_rouge(tok, logits, batch)
            rouge_l_f += result['rouge-l']['f']

    print(
        f'Epoch {epc} Test Rouge_l_f1: {rouge_l_f / dl_len}')  # , Test2 Rouge_l_f1: {rouge_l_f1_x / dl_len if attacker2 else 0}')
    p = get_inverter_path(sip_args, training_args, b2tr)
    if rouge_l_f / dl_len > training_args.save_threshold and training_args.save_checkpoint:
        print(f'Saving Model @ {p}epoch_{epc}_rouge_{rouge_l_f / dl_len:.4f}')
        attacker.save_pretrained(p + f'epoch_{epc}_rouge_{rouge_l_f / dl_len:.4f}')
    if training_args.log_to_wandb:
        log_dict = {'sip_training_epoch': epc, 'sip_test_rouge_l_f1': rouge_l_f / dl_len}
        wandb.log(log_dict)
    md.train(True)
    attacker.train(True)
    return rouge_l_f / dl_len


def pre_no_peek(noise_scale, llm, data_loader, max_steps=560):
    opt = AdamW([p for p in llm.parameters() if p.requires_grad], lr=1e-5)
    pbar = tqdm(total=max_steps)
    step = 0
    llm.train(True)
    with FLConfigHolder(llm) as ch:
        llm.fl_config.attack_mode = None
        llm.fl_config.collect_intermediates = True
        llm.fl_config.collect_all_layers = True
        ch.change_config()
        for epc in range(100):
            for batch in data_loader:
                opt.zero_grad()
                input_ids = batch['input_ids'].to(llm.device)
                attention_mask = batch['attention_mask'].to(llm.device)
                outputs = llm(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                b2tr, tr2t, all_inter = llm.get_all_inter(detach=False)
                embed = all_inter['embedding']
                dcor = dist_corr(embed.fx, b2tr.fx)
                loss += noise_scale * dcor
                loss.backward()
                opt.step()
                pbar.set_description(f'Loss {loss.item()}:.5f, DCOR {dcor.item():.5f}')
                pbar.update(1)
                step += 1
                if step >= max_steps:
                    break
            if step >= max_steps:
                break


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


def llm_forward(model, batch, tokenizer):
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


def experts_no_peek_pre(sip_arg: SIPAttackerArguments, expert_scales, llm, dataloader, random_num=15):
    save_dir = get_lora_path(sip_arg, 'dc')
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
        print(f'PK_NORM_{scale}:', calc_pk_norm(pk))
        result[scale] = pk
    if len(result) >= len([x for x in expert_scales if x >= 0]) + random_num:
        left_scales = set(result.keys()) - set(expert_scales)
        return result, list(left_scales)
    random_scales = []
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


def _resolve_noise_args(prefix):
    noise_scale = 0.0
    noise_mode = 'none'
    noise_prefixes = ['dxp', 'gaussian', 'dc']
    for np in noise_prefixes:
        if prefix.startswith(np):
            noise_mode = np
            if ':' in prefix:
                noise_scale = prefix[len(np) + 1:]
            break
    return noise_mode, noise_scale


def _get_data_loaders(sip_args: SIPAttackerArguments, training_args: InversionModelTrainingArgument, tokenizer):
    if ',' not in sip_args.dataset:
        dataset_cls = get_dataset_class(sip_args.dataset)
        dataset = dataset_cls(tokenizer, [])
        if get_dra_train_label(sip_args.dataset) == get_dra_test_label(sip_args.dataset):  # self-testing
            dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                                          type=get_dra_train_label(sip_args.dataset),
                                                                          shrink_frac=sip_args.train_frac,
                                                                          further_test_split=0.3)
        else:
            dataloader = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                         type=get_dra_train_label(sip_args.dataset),
                                                         shrink_frac=sip_args.train_frac)
            dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                              type=get_dra_test_label(sip_args.dataset),
                                                              shrink_frac=training_args.test_frac)
    else:
        dataset = MixtureFedDataset(tokenizer, [], sip_args.train_frac, sip_args.dataset.split(','),
                                    [get_dataset_class(nm) for nm in sip_args.dataset.split(',')])
        dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                                      type=None,
                                                                      shrink_frac=sip_args.train_frac)
    return dataloader, dataloader_test


def _get_output(tokenizer, text, encoder_model, attack_model):
    t = tokenizer(text, return_tensors="pt")
    inter = encoder_model(t['input_ids'].to(encoder_model.device),
                          attention_mask=t['attention_mask'].to(encoder_model.device))
    res = attack_model(inter)
    r = tokenizer.decode(res.argmax(dim=-1)[-1], skip_special_tokens=True)
    return r


def train_inversion_model(model, tokenizer, sip_args: SIPAttackerArguments, b2tr=True,
                          training_args: InversionModelTrainingArgument = None):
    if training_args is None:
        training_args = InversionModelTrainingArgument()
    layer = sip_args.b2tr_layer if b2tr else sip_args.tr2t_layer
    noise_mode, noise_scale = _resolve_noise_args(sip_args.prefix)
    config = FLConfig(collect_intermediates=False,
                      split_point_1=layer,
                      split_point_2=999,
                      attack_mode='b2tr',
                      noise_mode=noise_mode,
                      noise_scale=noise_scale,
                      use_lora_at_bottom=True,
                      use_lora_at_trunk=True,
                      use_lora_at_top=True
                      )

    if sip_args.prefix == 'nop':
        model.init_weights()  # reset params
    p = get_inverter_path(sip_args, training_args, b2tr)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p):
        print('Model exists, skipping...')
        return

    # model, tokenizer = get_model_and_tokenizer(sip_args.target_model_name, load_bits=sip_args.target_model_load_bits)
    dataloader, dataloader_test = _get_data_loaders(sip_args, training_args, tokenizer)
    reducer = None
    if sip_args.prefix.startswith('red'):
        tmp_args = {'model_name': sip_args.target_model_name, 'load_bits': sip_args.target_model_load_bits, 'dataset':
            sip_args.dataset}
        reducer = get_dim_reducer(argparse.Namespace(**tmp_args), model, tokenizer).to(model.device)
        config.reducer_enable = True

    model.config_sfl(config, dim_reducer=reducer)

    if noise_mode == 'dc':
        pre_no_peek(noise_scale, model, dataloader, max_steps=605)
    inverter_clz, cfg = get_inverter_with_config(sip_args.model)
    attack_model = inverter_clz(cfg, model.config, reduce_dim=None if reducer is None else reducer.config.alpha)
    # if not required_quantization(sip_args.target_model_name) and model.device == 'cpu':
    #     model.to(get_best_gpu())

    opt_cls = Adam
    if training_args.optim == 'adamw':
        opt_cls = AdamW
    optimizer = opt_cls(attack_model.parameters(), lr=training_args.lr, weight_decay=training_args.weight_decay)
    attack_model.to(model.device)
    # evaluate(0, model, attack_model, tokenizer, dataloader_test, args.save_dir)
    with tqdm(total=training_args.epochs * len(dataloader)) as pbar:
        for epc in range(training_args.epochs):
            model.train(True)
            rouge_l_f = 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                if noise_mode == 'dxp' and noise_scale < 0:
                    model.change_noise(random_choose_noise(sfl.config.dxp_moe_range))
                elif noise_mode == 'gaussian' and noise_scale < 0:
                    model.change_noise(random_choose_noise(sfl.config.gaussian_moe_range, mode='gaussian'))
                elif noise_mode == 'mix':
                    noise_mode = random.choice(['none', 'dxp', 'gaussian'])
                    if noise_mode == 'none':
                        model.change_noise(0, noise_mode)
                    elif noise_mode == 'dxp':
                        model.change_noise(random_choose_noise(dxp_moe_range, noise_mode), noise_mode)
                    else:
                        model.change_noise(random_choose_noise(gaussian_moe_range, mode='gaussian'), noise_mode)
                input_ids, intermediate = llm_forward(model, batch, tokenizer)
                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                loss.backward()
                optimizer.step()
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                rouge_l_f += res['rouge-l']['f']

                pbar.set_description(
                    f'Epoch {epc} Loss {loss.item():.5f}, Rouge_Lf1 {rouge_l_f / (step + 1):.4f}')
                if step % 300 == 0 and model.type != 'encoder-decoder':
                    q = "To mix food coloring with sugar, you can"
                    print(q, "==>", _get_output(tokenizer, q, model, attack_model))
                pbar.update(1)
                item_count += 1

            if (epc + 1) % training_args.checkpoint_freq == 0:
                evaluate(sip_args, training_args, b2tr, epc, model, attack_model, tokenizer, dataloader_test)
            if training_args.log_to_wandb:
                log_dict = {'sip_training_epoch': epc, 'sip_training_rouge_l_f1': rouge_l_f / item_count}
                wandb.log(log_dict)


def train_inversion_model_moe(model, tokenizer, sip_args: SIPAttackerArguments, b2tr=True,
                              training_args: InversionModelTrainingArgument = None
                              ):
    if training_args is None:
        training_args = InversionModelTrainingArgument()

    # model, tokenizer = get_model_and_tokenizer(sip_args.target_model_name, load_bits=sip_args.target_model_load_bits)
    layer = sip_args.b2tr_layer if b2tr else sip_args.tr2t_layer
    noise_mode, noise_scale = _resolve_noise_args(sip_args.prefix)

    def get_output(text, encoder_model, attack_model):
        t = tokenizer(text, return_tensors="pt")
        inter = encoder_model(t['input_ids'].to(model.device), attention_mask=t['attention_mask'].to(model.device))
        res = attack_model(inter)
        r = tokenizer.decode(res.argmax(dim=-1)[-1], skip_special_tokens=True)
        return r

    dataloader, dataloader_test = _get_data_loaders(sip_args, training_args, tokenizer)
    config = FLConfig(collect_intermediates=False,
                      split_point_1=layer,
                      split_point_2=999,
                      attack_mode='b2tr',
                      noise_mode=noise_mode,
                      use_lora_at_bottom=True,
                      use_lora_at_trunk=True,
                      use_lora_at_top=True
                      )
    model.config_sfl(config,
                     param_keeper=None)
    # model = model.convert_to_lora_model(restore_rest=False)

    expert_scales = [0, -1] + list(dxp_moe_range)
    if noise_mode == 'gaussian':
        expert_scales = [0, -1] + list(gaussian_moe_range)
    elif noise_mode == 'dc':
        expert_scales = [0, -1] + list(dc_moe_range)

    p = get_inverter_path(sip_args, training_args, b2tr)  # get_save_path(config, args.save_dir, args)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p):
        print('Model exists, skipping...')
        return
    extra_noise_scales = None
    nopeek_pks = None
    if noise_mode == 'dc':
        nopeek_pks, random_scales = experts_no_peek_pre(sip_args, expert_scales, model, dataloader)
        for scale, pk in nopeek_pks.items():
            print(f'NOPEEK {scale} {calc_pk_norm(pk)}')

        extra_noise_scales = random_scales

    # 开始训练Attack Model

    attack_model = MOEDRInverter(MOEDRAttackerConfig(expert_scales=expert_scales, hidden_size=256), model.config)
    #
    # if required_quantization(sip_args.target_model_name):
    #     device = get_best_gpu()
    #     model.to(device)

    # 阶段1， 不同专家独立训练解码能力
    attack_model.freeze_parts(experts=False, freeze=True)  # freeze gating
    attack_model.freeze_parts(experts=True, freeze=False)  # freeze gating
    optimizer = Adam([p for p in attack_model.parameters() if p.requires_grad], lr=training_args.lr)
    attack_model.to(model.device)
    with tqdm(total=training_args.epochs * len(dataloader)) as pbar:
        for epc in range(training_args.epochs):
            model.train(True)
            rougeLs = [0] * len(attack_model.config.expert_scales)
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                inters = []
                _, std_inter = llm_forward(model, batch, tokenizer)
                for noise_scale in attack_model.config.expert_scales:
                    if noise_scale < 0:
                        noise_scale = random_choose_noise(expert_scales, mode=noise_mode,
                                                          extra_choices=extra_noise_scales)
                    if noise_mode == 'dc':
                        with ParamRestored(llm=model, param_keeper=nopeek_pks[noise_scale],
                                           parts=['bottom', 'trunk', 'top'],
                                           write_back=False, disable_inter_collection=False):
                            input_ids, intermediate = llm_forward(model, batch, tokenizer)
                    else:
                        model.change_noise(noise_scale)
                        input_ids, intermediate = llm_forward(model, batch, tokenizer)
                    inters.append(intermediate)
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
            if training_args.log_to_wandb:
                log_dict = {'epoch': epc, }
                for i, rgl in enumerate(rougeLs):
                    log_dict.update({f"Expert{i}_RgL": rgl / item_count})
                wandb.log(log_dict)

    # 阶段2，训练Gating
    attack_model.freeze_parts(experts=True, freeze=False)  # activate experts
    attack_model.freeze_parts(experts=False, freeze=False)  # activate gating
    optimizer2 = Adam([p for p in attack_model.parameters() if p.requires_grad], lr=training_args.lr)
    with tqdm(total=training_args.gating_epochs * len(dataloader)) as pbar:
        for epc in range(training_args.gating_epochs):
            model.train(True)
            rougeL_total = 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                random_noise = random_choose_noise(attack_model.config.expert_scales, mode=noise_mode,
                                                   extra_choices=extra_noise_scales)
                if noise_mode != 'dc':
                    model.change_noise(random_noise)
                    input_ids, intermediate = llm_forward(model, batch, tokenizer)
                else:
                    with ParamRestored(llm=model, param_keeper=nopeek_pks[random_noise],
                                       parts=['bottom', 'trunk'],
                                       write_back=False, disable_inter_collection=False):
                        input_ids, intermediate = llm_forward(model, batch, tokenizer)
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
        if (epc + 1) % training_args.checkpoint_freq == 0:
            evaluate(sip_args, training_args, b2tr, epc, model, attack_model, tokenizer, dataloader_test)
        if training_args.log_to_wandb:
            log_dict = {'epoch': epc, 'Total_RgL': rougeL_total / item_count}
            wandb.log(log_dict)

    # 阶段3，训练整体模型
    attack_model.freeze_parts(experts=True, freeze=False)  # freeze experts
    attack_model.freeze_parts(experts=False, freeze=False)  # freeze gating
    optimizer3 = Adam([p for p in attack_model.parameters() if p.requires_grad], lr=training_args.lr)
    with tqdm(total=training_args.ft_epochs * len(dataloader)) as pbar:
        for epc in range(training_args.ft_epochs):
            model.train(True)
            rougeL_total = 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer2.zero_grad()
                random_noise = random_choose_noise(attack_model.config.expert_scales, mode=noise_mode,
                                                   extra_choices=extra_noise_scales)
                if noise_mode != 'dc':
                    model.change_noise(random_noise)
                    input_ids, intermediate = llm_forward(model, batch, tokenizer)
                else:
                    with ParamRestored(llm=model, param_keeper=nopeek_pks[random_noise],
                                       parts=['bottom', 'trunk'],
                                       write_back=False, disable_inter_collection=False):
                        input_ids, intermediate = llm_forward(model, batch, tokenizer)

                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                rougeL_total += res['rouge-l']['f']
                loss.backward()
                optimizer3.step()

                pbar.set_description(
                    f'FINAL Epoch {epc} Loss {loss.item():.5f}, RougeL {rougeL_total / (step + 1):.4f}')
                if step % 300 == 0 and model.type != 'encoder-decoder':
                    q = "To mix food coloring with sugar, you can"
                    print(q, "==>", get_output(q, model, attack_model))
                pbar.update(1)
                item_count += 1
        # 计算测试集上的ROGUE
        if (epc + 1) % training_args.checkpoint_freq == 0:
            evaluate(sip_args, training_args, b2tr, epc * 100, model, attack_model, tokenizer, dataloader_test)
        if training_args.log_to_wandb:
            log_dict = {'sip_training_epoch': epc, 'sip_training_rouge_l_f1': rougeL_total / item_count}
            wandb.log(log_dict)
