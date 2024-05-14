import argparse
import os
import sys

import torch
import wandb
from tqdm import tqdm
from transformers import AdamW

sys.path.append(os.path.abspath('../../..'))
from sfl.simulator.dataset import MixtureFedDataset

from sfl.config import FLConfig
from sfl.model.attacker.sip_attacker import ViTDRAttacker, ViTDRAttackerConfig
from sfl.utils.exp import get_model_and_tokenizer, get_dataset_class, add_train_dra_params
from sfl.utils.model import get_best_gpu, set_random_seed, \
    evaluate_attacker_mse

from sfl.config import DRA_train_label, DRA_test_label


def get_save_path(fl_config, save_dir, args):
    model_name = args.model_name
    cut_layer = fl_config.split_point_1 if fl_config.attack_mode == "b2tr" else fl_config.split_point_2
    if ',' in args.dataset:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/Tr{args.dataset_train_frac:.3f}-Ts*{args.dataset_test_frac:.3f}'
                         f'/{args.attack_model}/layer{cut_layer}/')
    else:
        p = os.path.join(save_dir,
                         f'{model_name}/{args.dataset}/{DRA_train_label[args.dataset]}*{args.dataset_train_frac:.3f}-{DRA_test_label[args.dataset]}*{args.dataset_test_frac:.3f}'
                         f'/{args.attack_model}/layer{cut_layer}/')
    attacker_prefix = 'normal/'
    if fl_config.noise_mode == 'gaussian':
        attacker_prefix = f'{fl_config.noise_mode}:{fl_config.noise_scale_gaussian}/'
    p += attacker_prefix
    return p


def evaluate(epc, md, attacker, test_data_loader, args):
    """
    恢复的评价指标选用MSE
    :return: MSE
    """
    md.eval()
    attacker.eval()
    dl_len = len(test_data_loader)
    with torch.no_grad():
        avg_mse = 0
        for step, batch in tqdm(enumerate(test_data_loader), total=dl_len):
            input_tensor = batch['input'].to(md.device)
            inter = md(input_tensor)
            logits = attacker(inter)
            avg_mse += evaluate_attacker_mse(logits, input_tensor)
    print(
        f'Epoch {epc} Test MSE: {avg_mse / dl_len}')
    p = get_save_path(md.fl_config, args.save_dir, args)
    if args.save_checkpoint:
        attacker.save_pretrained(p + f'epoch_{epc}_mse_{avg_mse / dl_len:.4f}')
    if args.log_to_wandb:
        log_dict = {'epoch': epc, 'test_mse_1': avg_mse / dl_len}
        wandb.log(log_dict)
    md.train(True)
    attacker.train(True)
    return avg_mse / dl_len


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

    model, processor = get_model_and_tokenizer(args.model_name, load_bits=args.load_bits)
    if ',' not in args.dataset:
        dataset_cls = get_dataset_class(args.dataset)
        dataset = dataset_cls(processor, [])
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
        dataset = MixtureFedDataset(processor, [], args.dataset_train_frac, args.dataset.split(','),
                                    [get_dataset_class(nm) for nm in args.dataset.split(',')])
        dataloader, dataloader_test = dataset.get_dataloader_unsliced(batch_size=args.batch_size,
                                                                      type=None,
                                                                      shrink_frac=args.dataset_train_frac)

    model.config_sfl(config,
                     param_keeper=None)
    # freeze all parts:
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 开始训练Attack Model
    attack_model = ViTDRAttacker(ViTDRAttackerConfig(), model.config)
    device = get_best_gpu()
    model.to(device)
    optimizer = AdamW(attack_model.parameters(), lr=args.lr)
    attack_model.to(model.device)

    epoch = args.epochs
    if args.log_to_wandb:
        wandb.init(project=args.exp_name,
                   name=f"{args.model_name}_{args.split_point_1}",
                   config=vars(args)
                   )
    evaluate(0, model, attack_model, dataloader_test, args)
    with tqdm(total=epoch * len(dataloader)) as pbar:
        for epc in range(epoch):
            model.train(True)
            mse_avg = 0
            item_count = 0
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_tensor = batch['input'].to(device)
                if args.noise_mode == 'gaussian':
                    # add random noise to the input (image of size bs, 3, 224, 224)
                    # the noise ranges from -args.noise_scale_gaussian to args.noise_scale_gaussian
                    noise = torch.randn_like(input_tensor) * args.noise_scale_gaussian
                    input_tensor = input_tensor + noise

                inter = model(input_tensor)
                recovered = attack_model(inter)
                loss = torch.nn.functional.mse_loss(recovered, input_tensor)
                loss.backward()
                optimizer.step()
                # 计算训练的MSE
                mse_avg += loss.item()

                pbar.set_description(
                    f'Epoch {epc} Loss {loss.item():.5f}, Avg MSE {mse_avg / (item_count + 1):.5f}')
                pbar.update(1)
                item_count += 1

            # 计算测试集上的ROGUE
            if (epc + 1) % args.checkpoint_freq == 0:
                evaluate(epc, model, attack_model, dataloader_test, args)
            if args.log_to_wandb:
                log_dict = {'epoch': epc,
                            'train_mse': mse_avg / item_count}
                wandb.log(log_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_dra_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    train_attacker(args)
