from abc import ABC, abstractmethod

import regex
from torch import nn

from sfl.simulator.param_keeper import ParameterKeeper
from sfl.utils import FLConfig


class SplitModel(nn.Module, ABC):
    """
    用于模拟三块切分SFL的模型。需要实现对top-trunk-bottom三个区域参数的读取与加载，以及切分位置前向传播和反向传播中间结果的读取
    """

    def __init__(self):
        super(SplitModel, self).__init__()
        self.param_keeper: ParameterKeeper | None = None
        self.fl_config: FLConfig | None = None

    def config_sfl(self, config: FLConfig, param_keeper: ParameterKeeper):
        self.fl_config = config
        self.param_keeper = param_keeper

    def print_split_model(self):
        print(f'=================Split-{self.config._name_or_path}=================')

        def group(named_params):
            ret = {}
            for nm, p in named_params:
                find = regex.search('\.[0-9]+\.', nm)
                if find:
                    find = find.span()[1]
                    prefix = nm[:find - 1]
                    suffix = nm[find:]
                else:
                    prefix = nm
                    suffix = ''
                ret.setdefault(prefix, [])
                ret[prefix].append((suffix, p.size()))
            return ret

        def print_group(g):
            for k, v in g.items():
                modules = ', '.join([f'{nm}: {tuple(sz)}' for nm, sz in v])
                print(f'\n{k}:[{modules}]')

        print('==================Top Layers==================')
        top_group = group(self.get_top_params())
        print_group(top_group)
        print('==================Trunk Layers==================')
        trunk_group = group(self.get_trunk_params())
        print_group(trunk_group)
        print('==================Bottom Layers==================')
        bottom_group = group(self.get_bottom_params())
        print_group(bottom_group)
        print('=============================================')

    def load_top_params(self, params):
        for (nm, p), p1 in zip(self.get_top_params(), params):
            p.data = p1.data

    def load_bottom_params(self, params):
        for (nm, p), p1 in zip(self.get_bottom_params(), params):
            p.data = p1.data

    def load_trunk_params(self, params):
        for (nm, p), p1 in zip(self.get_trunk_params(), params):
            p.data = p1.data

    @abstractmethod
    def get_bottom_params(self):
        raise NotImplementedError

    @abstractmethod
    def get_top_params(self):
        raise NotImplementedError

    @abstractmethod
    def get_trunk_params(self):
        raise NotImplementedError

    @abstractmethod
    def get_bottom_to_trunk_fx(self):
        raise NotImplementedError

    @abstractmethod
    def get_trunk_to_top_fx(self):
        raise NotImplementedError

    @abstractmethod
    def get_top_to_trunk_grad(self):
        raise NotImplementedError

    @abstractmethod
    def get_trunk_to_bottom_grad(self):
        raise NotImplementedError

    @abstractmethod
    def _store_bottom_to_trunk_fx(self, fx):
        raise NotImplementedError

    @abstractmethod
    def _store_trunk_to_top_fx(self, fx):
        raise NotImplementedError
