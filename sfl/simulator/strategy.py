import abc
from abc import ABC
from copy import deepcopy
from typing import Any, Iterator

from sfl.config import FLConfig, Intermediate
from sfl.model.split_model import SplitModel


class FLStrategy(ABC):
    """
    定义联邦学习的关键策略
    """

    def __init__(self, simulator=None):
        self.simulator = simulator
        self.client_logs = {}
        self.task_type = 'lm'

    @abc.abstractmethod
    def client_step(self, client_id: str, global_round, client_epoch, model: SplitModel, iterator: Iterator,
                    config: FLConfig):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_intermediate_result(self, global_round, client_id, local_epoch, local_step,
                                     b2tr_inter: Intermediate, tr2t_inter: Intermediate,
                                     all_inter: dict[int, Intermediate],
                                     batch, logs):
        raise NotImplementedError

    @abc.abstractmethod
    def client_evaluate(self, global_round, client_id, log):
        raise NotImplementedError

    def aggregation_step(self, global_round, params: dict[str, Any]):
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

    def step_done(self, client_id, mini_step, batch, logs=None):
        self.simulator._client_one_step_done(client_id, mini_step, batch, logs)
