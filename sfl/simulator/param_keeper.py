from abc import ABC
from copy import deepcopy


class ParameterKeeper(ABC):
    """
    用于在内存/磁盘中维护维护Client/Server的模型参数
    """
    def __init__(self, client_ids: list[str]):
        self.client_ids = client_ids

    def get_client_params(self, client_id):
        raise NotImplementedError

    def store_client_params(self, client_id=None, params=None):
        raise NotImplementedError

    def get_server_params(self, client_id):
        raise NotImplementedError

    def store_server_params(self, client_id=None, params=None):
        raise NotImplementedError


class InMemoryParameterKeeper(ParameterKeeper):

    def __init__(self, client_ids: list[str]):
        super().__init__(client_ids)
        self.client_params = {}
        self.server_params = {}
        self.intermediate_params = {}

    def get_client_params(self, client_id):
        return self.client_params[client_id]

    def store_client_params(self, client_id=None, params=None):
        if not client_id:
            for cid in self.client_ids:
                self.client_params[cid] = deepcopy(params)
        else:
            self.client_params[client_id] = params

    def get_server_params(self, client_id):
        return self.server_params[client_id]

    def store_server_params(self, client_id=None, params=None):
        if not client_id:
            for cid in self.client_ids:
                self.server_params[cid] = deepcopy(params)
        else:
            self.server_params[client_id] = params

    def get_intermediate_params(self, type, id):
        return self.intermediate_params[type][id]

    def store_intermediate_params(self, type, id, params):
        self.intermediate_params.setdefault(type, {})
        self.intermediate_params[type][id] = params
