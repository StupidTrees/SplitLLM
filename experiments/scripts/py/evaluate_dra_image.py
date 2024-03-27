import os
import sys
from typing import Any, Iterator

import wandb
from tqdm import tqdm
from transformers import AdamW

sys.path.append(os.path.abspath('../../..'))

from sfl.simulator.strategy import BaseSFLStrategy

from sfl.utils.model import Intermediate, set_random_seed, evaluate_attacker_mse
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *


# 定义Client本地学习策略
class ImageFLStrategy(BaseSFLStrategy):

    def client_step(self, client_id: str, global_round, client_epoch, llm: SplitWrapperModel, iterator: Iterator,
                    config: FLConfig):
        optimizer = AdamW(llm.parameters(), lr=1e-3)
        avg_loss = 0
        batch_num = 0
        with tqdm(total=config.client_steps) as pbar:
            for step, batch in enumerate(iterator):
                optimizer.zero_grad()
                input_tensors = batch['input'].to(llm.device)
                labels = batch['labels'].to(llm.device)
                outputs = llm(input_tensors, labels=labels)

                loss = outputs.loss
                pbar.set_description(
                    f'Client {client_id} Epoch {client_epoch} Step {self.simulator.get_current_step(client_id, step)} Loss {loss.item():.3f}')
                avg_loss += loss.detach().cpu().item()
                loss.backward()

                batch_num += 1
                optimizer.step()
                self.step_done(client_id, step, batch,
                               {"step_loss": loss.detach().cpu().item(), "avg_loss": float(avg_loss / batch_num)})
                pbar.update(1)
                if 0 < config.max_global_step <= self.simulator.get_current_step(client_id, step)[1] + 1:
                    break

    def sample_attacker_triggered(self, global_round, client_id, local_epoch, local_step,
                                  b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                  all_inter: dict[Any, Intermediate],
                                  batch, logs):
        with torch.no_grad():
            for type, atk in zip(['tr2t', 'b2tr'], [self.dra2, self.dra1]):
                if atk is None:
                    continue
                atk.to(self.simulator.device)
                inter = b2tr_inter if type == 'b2tr' else tr2t_inter
                attacked = atk(inter.fx.to(atk.device))
                mse_res = evaluate_attacker_mse(attacked, batch['input'].to(atk.device)).cpu()
                self.log_to_sample_result(client_id, f'DRA_{type}_MSE', mse_res)
                self.log_to_all_result(client_id, f'DRA_{type}_MSE', mse_res)
                logs[f'DRA_{type}_MSE_step'] = mse_res


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)

    # 加载TAG攻击模型
    model.config_sfl(config, None)
    model.train()
    # 加载DRA攻击模型
    attacker, attacker2 = get_dra_attacker(get_dra_config(args))

    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, args.test_data_label,
                                                       shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=ImageFLStrategy(args, model, tokenizer, test_loader, attacker, attacker2, None),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config, args=args)
    # 加载Pre-FT数据集
    if args.pre_ft_dataset is not None and len(args.pre_ft_dataset) > 0:
        pre_ft_dataset = get_dataset(args.pre_ft_dataset, tokenizer=tokenizer, client_ids=[])
        pre_ft_loader = pre_ft_dataset.get_dataloader_unsliced(args.batch_size, args.pre_ft_data_label)
        simulator.pre_ft(pre_ft_loader, ['bottom', 'top'], max_steps=args.pre_ft_max_steps)

    wandb.init(
        project=args.exp_name,
        name=args.case_name,
        config=args
    )
    model.train()
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
