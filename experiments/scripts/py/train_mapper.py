import argparse
import os
import sys

import torch
import wandb
from torch.optim import Adam, AdamW
from tqdm import tqdm

sys.path.append(os.path.abspath('../../..'))
from sfl.data.base import MixtureFedDataset
from sfl.config import FLConfig
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, add_train_mapper_params, required_quantization
from sfl.utils.model import get_best_gpu, set_random_seed
from sfl.model.attacker.eia_attacker import LMMapper, LMMapperConfig

from sfl.config import DRA_train_label, DRA_test_label


def get_save_path(save_dir, args):
    model_name = args.model_name
    if required_quantization(args.model_name):
        model_name += f'-{args.load_bits}bits'
    p = os.path.join(save_dir,
                     f'{model_name}/{args.dataset}/{DRA_train_label[args.dataset]}*{args.dataset_train_frac:.3f}-{DRA_test_label[args.dataset]}*{args.dataset_test_frac:.3f}'
                     f'/{args.target}/')
    return p


def evaluate(epc, md, mapper, test_data_loader, args):
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
    p = get_save_path(args.save_dir, args)
    if mase_avg / dl_len < args.save_threshold and args.save_checkpoint:
        mapper.save_pretrained(p + f'epoch_{epc}_mse_{mase_avg / dl_len:.6f}')
    if args.log_to_wandb:
        log_dict = {'epoch': epc, 'test_mse': mase_avg / dl_len}
        wandb.log(log_dict)
    md.train(True)
    mapper.train(True)
    return mase_avg


def train_mapper(args):
    """
    Train the mapper, which maps the deep layer's intermediate representations to the shallow layer's intermediate
    :param args:
    """
    config = FLConfig(collect_intermediates=True,
                      split_point_1=int(args.target.split('-')[1]),
                      split_point_2=int(args.target.split('-')[0]),
                      attack_mode='tr2t',
                      collect_all_layers=True
                      )

    p = get_save_path(args.save_dir, args)
    print(f'Checking Existing Model @ {p}')
    if os.path.exists(p) and args.skip_exists:
        print('Model exists, skipping...')
        return

    model, tokenizer = get_model_and_tokenizer(args.model_name)  # , load_bits=args.load_bits, force_on_best_gpu=True)
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

    model.config_sfl(config, param_keeper=None)
    # # freeze all parts:
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    mapper = LMMapper(LMMapperConfig(structure='linear', n_layers=2), target_config=model.config)
    if not hasattr(model.config, 'quantization_config') and model.device == 'cpu':
        model.to(get_best_gpu())

    opt_cls = Adam
    if args.opt == 'adamw':
        opt_cls = AdamW
    optimizer = opt_cls(mapper.parameters(), lr=args.lr, weight_decay=args.wd)
    # mapper.to(model.device)

    epoch = args.epochs
    if args.log_to_wandb:
        wandb.init(project=args.exp_name,
                   name=f"{args.model_name}_{args.target}",
                   config=vars(args)
                   )
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
                if 'chatglm' in args.model_name:
                    mapped = mapper(inter_tr2t.fx.to(mapper.device).float().permute(1, 0, 2))
                    # loss = torch.nn.MSELoss(reduction='sum')(mapped, inter_b2tr.fx.to(mapper.device).float().permute(1, 0, 2))
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
            if (epc + 1) % args.checkpoint_freq == 0:
                evaluate(epc, model, mapper, dataloader_test, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_mapper_params(parser)
    args, _ = parser.parse_known_args()
    set_random_seed(args.seed)
    train_mapper(args)
