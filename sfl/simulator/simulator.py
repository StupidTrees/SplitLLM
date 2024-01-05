import random

import torch

from sfl.config import FLConfig
from sfl.model.split_model import SplitModel
from sfl.simulator.dataset import FedDataset
from sfl.simulator.param_keeper import InMemoryParameterKeeper
from sfl.simulator.strategy import FLStrategy
from sfl.utils import get_best_gpu, tensor_bytes, size_str


class SFLSimulator(object):
    """
    SFL实验模拟
    """

    def __init__(self, client_ids, strategy: FLStrategy, llm: SplitModel, tokenizer, dataset: FedDataset,
                 config: FLConfig):
        self.client_ids = client_ids
        self.strategy: FLStrategy = strategy
        self.strategy.simulator = self
        self.llm: SplitModel = llm
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.device = get_best_gpu() if torch.cuda.is_available() else 'cpu'
        self.parameter_keeper = InMemoryParameterKeeper(client_ids)
        self.llm.config_sfl(config, self.parameter_keeper)
        self.communication_overhead_uplink = {}
        self.communication_overhead_downlink = {}
        self.current_global_round = 0
        if config.use_lora_at_trunk and not self.llm.adapter_added:
            self.llm = self.llm.convert_to_lora_model()
        if config.top_and_bottom_from_scratch:
            self.llm.reset_params(self.llm.get_top_params())
            self.llm.reset_params(self.llm.get_bottom_params())

    def simulate(self):
        self.llm.to(self.device)
        self.llm.train()
        # initialize server parameters
        self.parameter_keeper.store_server_params('server',
                                                  [p.detach().cpu() for nm, p in self.llm.get_trunk_params()])
        # initialize client parameters
        cm = ([p.detach().cpu() for nm, p in self.llm.get_top_params()],
              [p.detach().cpu() for nm, p in self.llm.get_bottom_params()])
        self.parameter_keeper.store_client_params(None, cm)
        for i in range(self.config.global_round):
            self.current_global_round = i
            print(f'==================================Global Round {i}=================================')
            # sample clients
            sampled_clients = random.sample(self.client_ids, int(len(self.client_ids) * self.config.client_per_round))
            # sequentially train each client
            for client_id in sampled_clients:
                self._client_step(i, client_id)
                self.__summarize_communication(i, client_id)
            self.__summarize_communication(i, client_id=None)
            # aggregate server parameters
            self._server_step(i, sampled_clients)
        self.__summarize_communication()

    def _client_step(self, global_round, client_id: str):
        # load client parameters (bottom and top layers)
        cm = self.parameter_keeper.get_client_params(client_id)
        self.llm.load_top_params(cm[0])
        self.llm.load_bottom_params(cm[1])
        # load server parameters (trunk layers or adapters)
        self.llm.load_trunk_params(self.parameter_keeper.get_server_params('server'))
        # client step
        torch.cuda.empty_cache()
        self.llm.to(self.device)
        self.strategy.client_step(global_round, client_id, self.llm, self.dataset.get_dataloader(client_id), self.config)
        # store updated client parameters
        cm = ([p.cpu() for nm, p in self.llm.get_top_params()],
              [p.cpu() for nm, p in self.llm.get_bottom_params()])
        self.parameter_keeper.store_client_params(client_id, cm)
        # store updated server parameters
        self.parameter_keeper.store_server_params(client_id,
                                                  [p.detach().cpu() for nm, p in self.llm.get_trunk_params()])

    def _server_step(self, global_round, sample_clients):
        # aggregate server parameters
        params = {}
        for client_id in sample_clients:
            params[client_id] = self.parameter_keeper.get_server_params(client_id)
        self.parameter_keeper.store_server_params('server', self.strategy.aggregation_step(global_round,params))

    def __summarize_communication(self, global_round=None, client_id=None):
        if global_round is None:
            total_uplink = 0
            total_downlink = 0
            for gr in self.communication_overhead_uplink:
                for cid in self.communication_overhead_uplink[gr]:
                    total_uplink += sum(self.communication_overhead_uplink[gr][cid].values())
                    total_downlink += sum(self.communication_overhead_downlink[gr][cid].values())
            print(f'FL communication overhead: uplink={size_str(total_uplink)}, downlink={size_str(total_downlink)}')

        elif client_id is None:
            total_uplink = 0
            total_downlink = 0
            for cid in self.communication_overhead_uplink[global_round]:
                total_uplink += sum(self.communication_overhead_uplink[global_round][cid].values())
                total_downlink += sum(self.communication_overhead_downlink[global_round][cid].values())
            print(
                f'Global Round {global_round} communication overhead: uplink={size_str(total_uplink)}, downlink={size_str(total_downlink)}')
        else:
            print(
                f'Client {client_id} communication overhead: uplink:{size_str(sum(self.communication_overhead_uplink[global_round][client_id].values()))},'
                f' downlink:{size_str(sum(self.communication_overhead_downlink[global_round][client_id].values()))}')

    def _collect_fp_result(self, client_id, local_epoch, local_step, batch):
        """
        在这里拿到前传的传输数据
        """
        b2tr = self.llm.get_bottom_to_trunk_fx()  # bottom-to-trunk
        tr2t = self.llm.get_trunk_to_top_fx()  # trunk-to-top
        self.strategy.callback_fp_param(client_id, local_epoch, local_step, b2tr, tr2t, batch)

        self.communication_overhead_uplink.setdefault(self.current_global_round, {})
        self.communication_overhead_uplink[self.current_global_round].setdefault(client_id, {})
        self.communication_overhead_uplink[self.current_global_round][client_id].setdefault(local_epoch, 0)
        self.communication_overhead_uplink[self.current_global_round][client_id][local_epoch] += tensor_bytes(b2tr)
        self.communication_overhead_downlink.setdefault(self.current_global_round, {})
        self.communication_overhead_downlink[self.current_global_round].setdefault(client_id, {})
        self.communication_overhead_downlink[self.current_global_round][client_id].setdefault(local_epoch, 0)
        self.communication_overhead_downlink[self.current_global_round][client_id][local_epoch] += tensor_bytes(tr2t)
        # print(f'FP: bottom->trunk size={b2tr.size()}, trunk->top size={tr2t.size()}')

    def _collect_bp_result(self, client_id, local_epoch, local_step, batch):
        """
        在这里拿到反传的中间数据
        """
        t2tr = self.llm.get_top_to_trunk_grad()  # top-to-trunk
        tr2b = self.llm.get_trunk_to_bottom_grad()  # trunk-to-bottom
        self.strategy.callback_bp_param(client_id, local_epoch, local_step, t2tr, tr2b, batch)

        self.communication_overhead_uplink.setdefault(self.current_global_round, {})
        self.communication_overhead_uplink[self.current_global_round].setdefault(client_id, {})
        self.communication_overhead_uplink[self.current_global_round][client_id].setdefault(local_epoch, 0)
        self.communication_overhead_uplink[self.current_global_round][client_id][local_epoch] += tensor_bytes(t2tr)
        self.communication_overhead_downlink.setdefault(self.current_global_round, {})
        self.communication_overhead_downlink[self.current_global_round].setdefault(client_id, {})
        self.communication_overhead_downlink[self.current_global_round][client_id].setdefault(local_epoch, 0)
        self.communication_overhead_downlink[self.current_global_round][client_id][local_epoch] += tensor_bytes(tr2b)
        # print(f'BP: top->trunk size={t2tr.size()}, trunk->bottom size={tr2b.size()}')
