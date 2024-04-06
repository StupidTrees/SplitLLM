import os
import sys
from copy import deepcopy

import wandb
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath('../../..'))

from sfl.simulator.strategy import BaseSFLStrategy
from sfl.utils.model import set_random_seed
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.exp import *
from sklearn.manifold import TSNE


# 定义Client本地学习策略
class MultiLayerDRAFLStrategy(BaseSFLStrategy):
    """
    每一轮触发攻击：攻击每一层的中间输出
    """

    def client_evaluate(self, global_round, client_id, log):
        self.llm.train()
        cfg_bk = deepcopy(self.llm.fl_config)
        cfg: FLConfig = self.llm.fl_config
        cfg.collect_all_layers = True
        self.llm.config_sfl(cfg, self.llm.param_keeper, self.llm.b2tr_hooks)
        self.llm(self.sample_batch['input_ids'].to(self.llm.device)[:, :10],
                 attention_mask=self.sample_batch['input_att_mask'].to(self.llm.device)[:, :10])
        _, _, all_inter = self.llm.get_all_inter(detach=True)
        layer_ranges = {}
        all_tensors = []
        for idx, (key, inter) in enumerate(all_inter.items()):
            if inter.fx is not None:
                all_tensors.append(inter.fx)
                layer_ranges[key] = (len(all_tensors), len(all_tensors) + 1)
        # draw tsne using tensors
        all_tensors = torch.stack(all_tensors).view(len(all_tensors), -1)
        print(all_tensors.size())
        print('t-SNE-ing...')
        tsne = TSNE(n_components=2, learning_rate=100, random_state=501,
                    perplexity=min(30.0, len(all_tensors) - 1)).fit_transform(all_tensors)
        tsne_data = {cid: tsne[s:e] for cid, (s, e) in layer_ranges.items()}

        # draw scatter plot
        data = []
        for cid, dt in tsne_data.items():
            if len(dt) < 1:
                continue
            x = dt[:, 0].item()
            y = dt[:, 1].item()
            data.append((x, y, f'Layer {cid}'))
        table = wandb.Table(data=data, columns=["x", "y", "name"])
        wandb.log({f"layer_tsne_{client_id}": wandb.plot.scatter(table, "x", "y")})
        self.llm.config_sfl(cfg_bk, self.llm.param_keeper, self.llm.b2tr_hooks)


def sfl_with_attacker(args):
    model, tokenizer = get_model_and_tokenizer(args.model_name)
    # 配置联邦学习
    client_ids = [str(i) for i in range(args.client_num)]
    config = get_fl_config(args)
    # 加载数据集
    fed_dataset = get_dataset(args.dataset, tokenizer=tokenizer, client_ids=client_ids,
                              shrink_frac=args.data_shrink_frac)
    observe_sample = next(iter(fed_dataset.get_dataloader_unsliced(1, 'train')))
    test_dataset = get_dataset(args.dataset, tokenizer=tokenizer)
    test_loader = test_dataset.get_dataloader_unsliced(1, 'test', shrink_frac=args.test_data_shrink_frac)
    simulator = SFLSimulator(client_ids=client_ids,
                             strategy=MultiLayerDRAFLStrategy(args, model, tokenizer, test_loader, None, None,
                                                              sample_batch=observe_sample),
                             llm=model,
                             tokenizer=tokenizer,
                             dataset=fed_dataset, config=config)
    wandb.init(
        project=args.exp_name,
        name=args.case_name,
        config=args
    )
    simulator.simulate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_sfl_params(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    sfl_with_attacker(args)
