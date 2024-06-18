import os
import sys
from typing import Any, Iterator

import wandb
from tqdm import tqdm
from transformers import AdamW


sys.path.append(os.path.abspath('../../..'))
from sfl.model.attacker.sip_attacker import SIPAttacker
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.strategies.basic import BaseSFLStrategy

from sfl.utils.model import Intermediate, set_random_seed, evaluate_attacker_mse, convert_to_image, evaluate_accuracy, \
    evaluate_perplexity
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *


class ImageFLStrategy(BaseSFLStrategy):

    def __init__(self, args, llm, tokenizer, test_loader, attacker, sample_batch):
        super().__init__(args, llm, tokenizer, test_loader, attacker)
        self.sample_batch = sample_batch
        self.attacker = attacker

    def client_evaluate(self, global_round, client_id, log):
        if self.task_type == 'classification':
            log['test-acc'] = evaluate_accuracy(self.simulator.llm, self.test_loader)
        elif self.task_type == 'lm':
            ppl = evaluate_perplexity(self.simulator.llm, self.test_loader)
            log['test-ppl'] = ppl
        log_sample_image(self.llm, self.attacker, self.sample_batch)

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
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    model.config_sfl(config, None)
    model.train()

    # 加载DRA攻击模型
    sip_attacker = SIPAttacker()
    sip_attacker.load_attacker(args, sip_attacker.parse_arguments(args, 'sip'))
    attacker = sip_attacker.inverter_b2tr

    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=[])
    test_loader = test_dataset.get_dataloader_unsliced(args.batch_size, args.test_data_label,
                                                       shrink_frac=args.test_data_shrink_frac)
    sample_batch = next(iter(fed_dataset.get_dataloader_unsliced(3, 'train')))
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=ImageFLStrategy(args, model, tokenizer, test_loader, attacker, sample_batch),
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
    log_sample_image(model, attacker, sample_batch)
    model.train()
    simulator.simulate()


def log_sample_image(llm, attacker, observe_sample):
    cfg = llm.fl_config
    cfg.attack_mode = 'b2tr'
    llm.config_sfl(cfg, llm.param_keeper)
    input_tensors = observe_sample['input'].to(llm.device)
    inter = llm(input_tensors)
    attacked = attacker(inter.to(llm.device))
    images = convert_to_image(attacked)
    wandb.log({'attacked_sample_images': [wandb.Image(img) for img in images]})
    cfg = llm.fl_config
    cfg.attack_mode = None
    llm.config_sfl(cfg, llm.param_keeper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
