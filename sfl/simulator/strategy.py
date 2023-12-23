import abc
from abc import ABC
from copy import deepcopy
from typing import Any

from torch.utils.data import DataLoader

from sfl.model.split_model import SplitModel
from sfl.utils import FLConfig


class FLStrategy(ABC):
    """
    定义联邦学习的关键策略
    """

    def __init__(self, simulator=None):
        self.simulator = simulator

    @abc.abstractmethod
    def client_step(self, client_id: str, model: SplitModel, dataloader: DataLoader, config: FLConfig):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_fp_param(self, client_id, local_epoch, local_step, b2tr_params, tr2t_params, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_bp_param(self, client_id, local_epoch, local_step, t2tr_params, tr2b_params, batch):
        raise NotImplementedError

    def aggregation_step(self, params: dict[str, Any]):
        res = None
        for k, v in params.items():
            if res is None:
                res = deepcopy(v)
            else:
                for p, p1 in zip(res, v):
                    p.data += p1.data
        for p in res:
            p.data /= len(params)
        return res

    def fp_done(self, client_id, local_epoch, local_step,batch):
        self.simulator._collect_fp_result(client_id, local_epoch, local_step, batch)

    def bp_done(self, client_id, local_epoch, local_step, batch):
        self.simulator._collect_bp_result(client_id, local_epoch, local_step, batch)
