from abc import ABC, abstractmethod

import regex
from peft import LoraConfig, get_peft_model
from torch import nn

from sfl.config import FLConfig, Intermediate
from sfl.simulator.param_keeper import ParameterKeeper


class SplitModel(nn.Module, ABC):
    """
    用于模拟三块切分SFL的模型。需要实现对top-trunk-bottom三个区域参数的读取与加载，以及切分位置前向传播和反向传播中间结果的读取
    """

    def __init__(self):
        super().__init__()
        self.param_keeper: ParameterKeeper | None = None
        self.fl_config: FLConfig | None = None
        self.adapter_added = False
        self.b2tr_hooks = []
        self.perturber = None
        self.intermediate_fx = {}

    def config_sfl(self, config: FLConfig, param_keeper: ParameterKeeper | None, b2tr_hooks: list = None):
        self.fl_config = config
        self.param_keeper = param_keeper
        if b2tr_hooks is not None:
            for hk in b2tr_hooks:
                self.b2tr_hooks.append(hk)

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
            p.data = p1.data.to(self.device)

    def load_bottom_params(self, params):
        for (nm, p), p1 in zip(self.get_bottom_params(), params):
            if not p.requires_grad:
                continue
            p.data = p1.data.to(self.device)

    def load_trunk_params(self, params):
        for (nm, p), p1 in zip(self.get_trunk_params(), params):
            if not p.requires_grad:
                continue
            p.data = p1.data.to(self.device)

    def convert_to_lora_model(self, restore_rest=True):
        """
        为Trunk部分加上LoRA适配器
        :return:
         """
        if self.adapter_added:
            return self
        if (not self.fl_config.use_lora_at_top) and (not self.fl_config.use_lora_at_bottom) and (
                not self.fl_config.use_lora_at_trunk):
            return self

        lora_config = LoraConfig(target_modules=self.get_adapter_module_regex())
        res = get_peft_model(self, lora_config)
        if restore_rest:
            # PEFT会冻结所有模型参数，需要恢复其他部分
            if not self.fl_config.use_lora_at_trunk:
                for name, param in res.get_trunk_params(trainable_only=False):
                    param.requires_grad = True
            if not self.fl_config.use_lora_at_top:
                for name, param in res.get_top_params(trainable_only=False):
                    param.requires_grad = True
            if not self.fl_config.use_lora_at_bottom:
                for name, param in res.get_bottom_params(trainable_only=False):
                    param.requires_grad = True
        self.adapter_added = True
        return res

    def reset_params(self, named_params, reset_mode):
        pass

    @abstractmethod
    def get_adapter_module_regex(self):
        """
        获得Trunk部分要加上Adapter的Module名称的正则表达式
        :return:
        """
        raise NotImplementedError

    def get_bottom_params(self, trainable_only=True):
        for nm, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if self._get_block_num(nm) >= self.fl_config.split_point_1:
                break
            else:
                yield nm, p

    def get_top_params(self, trainable_only=True):
        trunk = False
        for nm, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if self._get_block_num(nm) >= self.fl_config.split_point_2:
                trunk = True
            if trunk:
                yield nm, p

    def get_trunk_params(self, trainable_only=True):
        for nm, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if self.fl_config.split_point_1 <= self._get_block_num(nm) < self.fl_config.split_point_2:
                yield nm, p

    def get_all_inter(self, detach=True):
        res = {}
        bt2tr = None
        tr2t = None
        for idx, v in self.intermediate_fx.items():
            inter = Intermediate(v.detach().cpu() if detach else v)
            if v.grad is not None:
                inter.grad = v.grad.clone().detach().cpu() if detach else v.grad
            if idx == self.fl_config.split_point_1 - 1:
                inter.type = 'b2tr'
                bt2tr = inter
            elif idx == self.fl_config.split_point_2 - 1:
                inter.type = 'tr2t'
                tr2t = inter
            res[idx] = inter
        return bt2tr, tr2t, res

    def _store_fx(self, layer_index, fx):
        self.intermediate_fx[layer_index] = fx
