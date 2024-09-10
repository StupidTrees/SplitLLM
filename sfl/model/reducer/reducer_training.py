import os

import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm

from sfl.utils.args import FLConfig
from sfl.data.base import MixtureFedDataset
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.reducer.args import ReducerArgument, ReductionTrainingArguments, DRConfig
from sfl.model.reducer.reducer_models import DimReduction
from sfl.utils.exp import get_dataset_class, get_dra_test_label, get_dra_train_label
from sfl.utils.model import FLConfigHolder


def _get_data_loaders(red_args: ReducerArgument, training_args: ReductionTrainingArguments, tokenizer):
    if ',' not in red_args.dataset:
        dataset_cls = get_dataset_class(red_args.dataset)
        dataset = dataset_cls(tokenizer, [])
        if get_dra_train_label(red_args.dataset) == get_dra_test_label(red_args.dataset):  # self-testing
            dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                                          type=get_dra_train_label(red_args.dataset),
                                                                          shrink_frac=red_args.train_frac,
                                                                          further_test_split=0.3)
        else:
            dataloader = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                         type=get_dra_train_label(red_args.dataset),
                                                         shrink_frac=red_args.train_frac)
            dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                              type=get_dra_test_label(red_args.dataset),
                                                              shrink_frac=training_args.test_frac)
    else:
        dataset = MixtureFedDataset(tokenizer, [], red_args.train_frac, red_args.dataset.split(','),
                                    [get_dataset_class(nm) for nm in red_args.dataset.split(',')])
        dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=training_args.batch_size,
                                                                      type=None,
                                                                      shrink_frac=red_args.train_frac)
    return dataloader, dataloader_test


def get_save_path(red_arg: ReducerArgument, training_args: ReductionTrainingArguments):
    model_name = red_arg.target_model
    if 'llama' in model_name or 'chatglm' in model_name:
        model_name += f'-{red_arg.target_model_load_bits}bits'
    p = os.path.join(red_arg.path,
                     f'{model_name}/{red_arg.dataset}/{red_arg.train_label}*{red_arg.train_frac:.3f}-{get_dra_test_label(red_arg.dataset)}*{training_args.test_frac:.3f}'
                     f'/layer{red_arg.layer}/{red_arg.alpha}/')
    return p


def evaluate(epc, md: SplitWrapperModel, reducer: DimReduction | None, test_data_loader, red_arg: ReducerArgument,
             training_arg: ReductionTrainingArguments):
    if reducer:
        reducer.eval()
    with FLConfigHolder(md) as ch:
        md.fl_config.collect_intermediates = True
        ch.change_config()
        dl_len = len(test_data_loader)
        mase_avg = 0

        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            input_ids = batch['input_ids'].to(md.device)
            attention_mask = batch['attention_mask'].to(md.device)
            md(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            _, _, all_inters = md.get_all_inter(detach=True)
            from_inter = all_inters.get(red_arg.layer - 1, None)
            reduced, recovered = reducer(from_inter.fx.to(reducer.device))
            mase_avg += torch.nn.MSELoss()(recovered, from_inter.fx.to(reducer.device).float()).detach().item()
        print(f'Epoch {epc} Test MSE: {mase_avg / dl_len}')
        avg_mse = mase_avg / dl_len
        p = get_save_path(red_arg, training_arg)

        if training_arg.save_checkpoint:
            reducer.save_pretrained(p + f'epoch_{epc}_mse_{avg_mse:.5f}')

    md.train(True)
    if reducer:
        reducer.train(True)
    return avg_mse


def train_reducer(model, tokenizer, red_arg: ReducerArgument, training_arg: ReductionTrainingArguments = None):
    if training_arg is None:
        training_arg = ReductionTrainingArguments()
    config = FLConfig(collect_intermediates=True,
                      split_point_1=red_arg.layer + 1,
                      split_point_2=27,
                      attack_mode='b2tr',
                      collect_all_layers=True,
                      reducer_enable=False
                      )

    p = get_save_path(red_arg, training_arg)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p):
        print('Model exists, skipping...')
        return

    reducer = DimReduction(DRConfig(layer=red_arg.layer, alpha=red_arg.alpha), target_config=model.config)
    model.config_sfl(config, param_keeper=None, dim_reducer=reducer)

    dataloader, dataloader_test = _get_data_loaders(red_arg, training_arg, tokenizer)

    opt_cls = Adam
    if training_arg.opt == 'adamw':
        opt_cls = AdamW
    optimizer = opt_cls(reducer.parameters(), lr=training_arg.lr, weight_decay=1e-5)
    reducer.to(model.device)
    epoch = training_arg.epochs
    with tqdm(total=epoch * len(dataloader)) as pbar:
        for epc in range(epoch):
            model.train(True)
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                model(input_ids=input_ids, attention_mask=attention_mask)
                _, _, all_inters = model.get_all_inter(detach=True)
                from_inter = all_inters.get(red_arg.layer - 1, None)
                reduced, recovered = reducer(from_inter.fx.to(reducer.device))
                loss = torch.nn.MSELoss()(recovered, from_inter.fx.to(reducer.device).float())
                loss.backward()
                optimizer.step()
                pbar.set_description(
                    f'Epoch {epc} Loss {loss.item():.5f}')
                pbar.update(1)
                item_count += 1

            if (epc + 1) % training_arg.checkpoint_freq == 0:
                evaluate(epc, model, reducer, dataloader_test, red_arg, training_arg)
