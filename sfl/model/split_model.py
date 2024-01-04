from abc import ABC, abstractmethod

import regex
from peft import LoraConfig, get_peft_model
from torch import nn

from sfl.simulator.param_keeper import ParameterKeeper
from sfl.config import FLConfig


class SplitModel(nn.Module, ABC):
    """
    用于模拟三块切分SFL的模型。需要实现对top-trunk-bottom三个区域参数的读取与加载，以及切分位置前向传播和反向传播中间结果的读取
    """

    def __init__(self):
        super().__init__()
        self.param_keeper: ParameterKeeper | None = None
        self.fl_config: FLConfig | None = None
        self.adapter_added = False

    def config_sfl(self, config: FLConfig, param_keeper: ParameterKeeper|None):
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
            if not p.requires_grad:
                continue
            p.data = p1.data

    def load_bottom_params(self, params):
        for (nm, p), p1 in zip(self.get_bottom_params(), params):
            if not p.requires_grad:
                continue
            p.data = p1.data

    def load_trunk_params(self, params):
        for (nm, p), p1 in zip(self.get_trunk_params(), params):
            if not p.requires_grad:
                continue
            p.data = p1.data

    def convert_to_lora_model(self, restore_top_bottom=True):
        """
        为Trunk部分加上LoRA适配器
        :return:
         """
        if self.adapter_added:
            return self
        lora_config = LoraConfig(target_modules=self.get_trunk_adapter_module_regex())
        res = get_peft_model(self, lora_config)
        if restore_top_bottom:
            # PEFT会冻结所有模型参数，需要恢复top和bottom部分
            for name, param in res.get_top_params(trainable_only=False):
                param.requires_grad = True
            for name, param in res.get_bottom_params(trainable_only=False):
                param.requires_grad = True
        self.adapter_added = True
        return res

    def reset_params(self, named_params):
        pass

    @abstractmethod
    def get_trunk_adapter_module_regex(self):
        """
        获得Trunk部分要加上Adapter的Module名称的正则表达式
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_bottom_params(self, trainable_only=True):
        raise NotImplementedError

    @abstractmethod
    def get_top_params(self, trainable_only=True):
        raise NotImplementedError

    @abstractmethod
    def get_trunk_params(self, trainable_only=True):
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

    def _store_bottom_to_trunk_fx(self, fx):
        raise NotImplementedError

    def _store_trunk_to_top_fx(self, fx):
        raise NotImplementedError
