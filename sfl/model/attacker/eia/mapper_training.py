import os
import sys

import torch
import wandb
from torch.optim import Adam, AdamW
from tqdm import tqdm

from sfl.model.attacker.eia.args import EIAArguments, MapperTrainingArguments
from sfl.model.attacker.eia.eia_attacker import LMMapper
from sfl.data.base import MixtureFedDataset
from sfl.utils.args import FLConfig
from sfl.model.attacker.eia.mapper_models import LMMapperConfig
from sfl.utils.exp import get_dataset_class, required_quantization
from sfl.utils.model import get_best_gpu
from sfl.utils.exp import get_dra_train_label, get_dra_test_label


def get_save_path(eia_args: EIAArguments, training_args: MapperTrainingArguments):
    model_name = eia_args.mapper_target_model_name
    if required_quantization(eia_args.mapper_target_model_name):
        model_name += f'-{eia_args.mapper_target_model_load_bits}bits'
    p = os.path.join(eia_args.mapper_path,
                     f'{model_name}/{eia_args.mapper_dataset}/{get_dra_train_label(eia_args.mapper_dataset)}*{eia_args.mapper_train_frac:.3f}-{get_dra_test_label(eia_args.mapper_dataset)}*{training_args.test_frac:.3f}'
                     f'/{eia_args.mapper_targets}/')
    return p


def evaluate(eia_args: EIAArguments, training_args: MapperTrainingArguments, epc, md, mapper, test_data_loader):
    mapper.eval()
    dl_len = len(test_data_loader)
    mase_avg = 0
    for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
        input_ids = batch['input_ids'].to(md.device)
        attention_mask = batch['attention_mask'].to(md.device)
        md(input_ids=input_ids, attention_mask=attention_mask)
        inter_b2tr, inter_tr2t, _ = md.get_all_inter(detach=True)
        mapped = mapper(inter_tr2t.fx.to(mapper.device))
        loss = torch.nn.MSELoss()(mapped, inter_b2tr.fx.to(mapper.device))
        mase_avg += loss.item()
    print(
        f'Epoch {epc} Test Rouge_l_f1: {mase_avg / dl_len}')  # , Test2 Rouge_l_f1: {rouge_l_f1_x / dl_len if attacker2 else 0}')
    p = get_save_path(eia_args, training_args)
    mapper.save_pretrained(p + f'epoch_{epc}_mse_{mase_avg / dl_len:.6f}')
    if training_args.log_to_wandb:
        log_dict = {'eia_mapper_training_epoch': epc, 'eia_mapper_test_mse': mase_avg / dl_len}
        wandb.log(log_dict)
    md.train(True)
    mapper.train(True)
    return mase_avg


def train_mapper(model, tokenizer, eia_args: EIAArguments, training_args: MapperTrainingArguments = None):
    if training_args is None:
        training_args = MapperTrainingArguments()

    config = FLConfig(collect_intermediates=True,
                      split_point_1=int(eia_args.mapper_targets.split('-')[1]),
                      split_point_2=int(eia_args.mapper_targets.split('-')[0]),
                      attack_mode='tr2t',
                      collect_all_layers=True
                      )

    p = get_save_path(eia_args, training_args)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p):
        print('Model exists, skipping...')
        return

    # model, tokenizer = get_model_and_tokenizer(
    #     eia_args.mapper_target_model_name)  # , load_bits=args.load_bits, force_on_best_gpu=True)
    if ',' not in eia_args.mapper_dataset:
        dataset_cls = get_dataset_class(eia_args.mapper_dataset)
        dataset = dataset_cls(tokenizer, [])
        if get_dra_train_label(eia_args.mapper_dataset) == get_dra_test_label(eia_args.mapper_dataset):  # self-testing
            dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                                          type=get_dra_train_label(eia_args.mapper_dataset),
                                                                          shrink_frac=eia_args.mapper_train_frac,
                                                                          further_test_split=0.3)
        else:
            dataloader = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                         type=get_dra_train_label(eia_args.mapper_dataset),
                                                         shrink_frac=eia_args.mapper_train_frac)
            dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                              type=get_dra_test_label(eia_args.mapper_dataset),
                                                              shrink_frac=training_args.test_frac)
    else:
        dataset = MixtureFedDataset(tokenizer, [], eia_args.mapper_train_frac, eia_args.mapper_dataset.split(','),
                                    [get_dataset_class(nm) for nm in eia_args.mapper_dataset.split(',')])
        dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                                      type=None,
                                                                      shrink_frac=eia_args.mapper_train_frac)

    model.config_sfl(config, param_keeper=None)
    mapper = LMMapper(LMMapperConfig(structure='linear', n_layers=2), target_config=model.config)
    if not hasattr(model.config, 'quantization_config') and model.device == 'cpu':
        model.to(get_best_gpu())

    opt_cls = Adam
    if training_args.opt == 'adamw':
        opt_cls = AdamW
    optimizer = opt_cls(mapper.parameters(), lr=training_args.lr, weight_decay=training_args.wd)
    # mapper.to(model.device)
    epoch = training_args.epochs
    with tqdm(total=epoch * len(dataloader)) as pbar:
        for epc in range(epoch):
            model.train(True)
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                model(input_ids=input_ids, attention_mask=attention_mask)
                inter_b2tr, inter_tr2t, _ = model.get_all_inter(detach=True)
                if 'chatglm' in eia_args.mapper_target_model_name:
                    mapped = mapper(inter_tr2t.fx.to(mapper.device).float().permute(1, 0, 2))
                    loss = 0
                    for x, y in zip(mapped, inter_b2tr.fx.to(mapper.device).float().permute(1, 0, 2)):
                        loss += (x - y).pow(2).sum()
                else:
                    mapped = mapper(inter_tr2t.fx.float().to(mapper.device))
                    loss = 0
                    for x, y in zip(mapped, inter_b2tr.fx.float().to(mapper.device)):
                        loss += (x - y).pow(2).sum()

                loss.backward()
                optimizer.step()
                pbar.set_description(
                    f'Epoch {epc} Loss {loss.item():.5f}')
                pbar.update(1)
                item_count += 1
            if (epc + 1) % training_args.checkpoint_freq == 0:
                evaluate(eia_args, training_args, epc, model, mapper, dataloader_test)
