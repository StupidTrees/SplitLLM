import argparse
import os
import sys

import torch
import wandb
from torch.optim import Adam, AdamW
from tqdm import tqdm

sys.path.append(os.path.abspath('../../..'))
from sfl.model.llm.dim_reduction import DimReduction, DRConfig
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.dataset import MixtureFedDataset
from sfl.config import FLConfig
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, add_train_reducer_params
from sfl.utils.model import get_best_gpu, set_random_seed, FLConfigHolder

from sfl.config import DRA_train_label, DRA_test_label


def get_save_path(save_dir, args):
    model_name = args.model_name
    if 'llama' in model_name or 'chatglm' in model_name:
        model_name += f'-{args.load_bits}bits'
    p = os.path.join(save_dir,
                     f'{model_name}/{args.dataset}/{args.dataset_train_label}*{args.dataset_train_frac:.3f}-{DRA_test_label[args.dataset]}*{args.dataset_test_frac:.3f}'
                     f'/layer{args.layer}/{args.alpha}/')
    return p


def evaluate(epc, md: SplitWrapperModel, reducer: DimReduction | None, test_data_loader, raw_ppl=None, args=None):
    """
    恢复的评价指标选用ROUGE
    :return: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-L-P, ROUGE-L-R
    """
    if reducer:
        reducer.eval()
    with FLConfigHolder(md) as ch:
        md.fl_config.attack_mode = None
        md.fl_config.collect_intermediates = False
        md.fl_config.reducer_enable = True
        ch.change_config()
        dl_len = len(test_data_loader)
        mase_avg = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
                input_ids = batch['input_ids'].to(md.device)
                attention_mask = batch['input_att_mask'].to(md.device)
                out = md(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                mase_avg += out.loss.detach().cpu().item()

        print(
            f'Epoch {epc} Test PPL: {mase_avg / dl_len}')
        avg_ppl = mase_avg / dl_len
        p = get_save_path(args.save_dir, args)
        if raw_ppl:
            ppl_frac = avg_ppl / raw_ppl
            if ppl_frac < args.save_threshold and args.save_checkpoint:
                reducer.save_pretrained(p + f'epoch_{epc}_rppl_{ppl_frac:.5f}')

    md.train(True)
    if reducer:
        reducer.train(True)
    return avg_ppl


def train_mapper(args):
    """
    训练层间映射
    :param args:
    """
    config = FLConfig(collect_intermediates=True,
                      split_point_1=args.layer + 1,
                      split_point_2=27,
                      attack_mode='b2tr',
                      collect_all_layers=True,
                      reducer_enable=False
                      )

    p = get_save_path(args.save_dir, args)
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

    # # freeze all parts:
    # for name, param in model.named_parameters():
    #     param.requires_grad = False

    # 开始训练Reducer
    reducer = DimReduction(DRConfig(layer=args.layer, alpha=args.alpha), target_config=model.config)
    model.config_sfl(config, param_keeper=None, dim_reducer=reducer)
    if 'llama' not in args.model_name and 'chatglm' not in args.model_name and 'vicuna' not in args.model_name:
        device = get_best_gpu()
        model.to(device)

    opt_cls = Adam
    if args.opt == 'adamw':
        opt_cls = AdamW
    optimizer = opt_cls(reducer.parameters(), lr=args.lr, weight_decay=1e-5)
    reducer.to(model.device)

    epoch = args.epochs
    if args.log_to_wandb:
        wandb.init(project=args.exp_name,
                   name=f"{args.model_name}_{args.target}",
                   config=vars(args)
                   )
    raw_ppl = evaluate(0, model, None, dataloader_test, None, args)
    print(f'Raw PPL:{raw_ppl:.6f}')
    # evaluate(0, model, attack_model, tokenizer, dataloader_test, args.save_dir)
    with tqdm(total=epoch * len(dataloader)) as pbar:
        for epc in range(epoch):
            model.train(True)
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['input_att_mask'].to(model.device)
                b2tr_hidden = model(input_ids=input_ids, attention_mask=attention_mask)
                _, _, all_inters = model.get_all_inter(detach=True)
                from_inter = all_inters.get(args.layer - 1, None)
                reduced, recovered = reducer(from_inter.fx.to(reducer.device))
                loss = torch.nn.MSELoss()(recovered, b2tr_hidden.to(reducer.device).float())
                loss.backward()
                optimizer.step()
                pbar.set_description(
                    f'Epoch {epc} Loss {loss.item():.5f}')
                pbar.update(1)
                item_count += 1

            # 计算测试集上的ppl
            if (epc + 1) % args.checkpoint_freq == 0:
                evaluate(epc, model, reducer, dataloader_test, raw_ppl, args)
            # if args.log_to_wandb:
            #     log_dict = {'epoch': epc
            #                 }
            #     wandb.log(log_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_reducer_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    train_mapper(args)
