from abc import ABC, abstractmethod

import regex
import torch
from peft import LoraConfig, get_peft_model
from torch import nn, float16

from sfl.config import FLConfig
from sfl.model.llm.dim_reduction import DimReduction
from sfl.model.llm.noise import GaussianPerturber
from sfl.simulator.param_keeper import ParameterKeeper
from sfl.utils.model import Intermediate


class SplitModel(nn.Module, ABC):
    """
    用于模拟三块切分SFL的模型。需要实现对切分位置前向传播和反向传播中间结果的存取
    """

    def __init__(self):
        super().__init__()
        self.param_keeper: ParameterKeeper | None = None
        self.fl_config: FLConfig | None = None
        self.adapter_added = False
        self.b2tr_hooks = []
        self.intermediate_fx = {}
        self.noise_mode = None
        self.dim_reducer = None
        self.perturbers = {'gaussian': GaussianPerturber()}

    def config_sfl(self, config: FLConfig, param_keeper: ParameterKeeper | None = None, b2tr_hooks: list = None,
                   dim_reducer: DimReduction = None, *args, **kwargs):
        self.fl_config = config
        self.noise_mode = config.noise_mode
        if dim_reducer is not None:
            self.dim_reducer = dim_reducer
        if param_keeper is not None:
            self.param_keeper = param_keeper
        if b2tr_hooks is not None:
            for hk in b2tr_hooks:
                self.b2tr_hooks.append(hk)
        if config.noise_mode == 'gaussian':
            self.perturbers['gaussian'].change_noise_scale(config.noise_scale_gaussian)

    def change_noise(self, noise_scale, noise_mode=None):
        if noise_mode is not None:
            self.noise_mode = noise_mode
        for k, v in self.perturbers.items():
            v.change_noise_scale(noise_scale)

    def get_all_inter(self, detach=True):
        res = {}
        bt2tr = None
        tr2t = None
        for idx, v in self.intermediate_fx.items():
            inter = Intermediate(v.detach().cpu() if detach and v is not None else v)
            if v is not None and v.grad is not None:
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

    def inject_after_embedding(self, inputs_embeds):
        if self.noise_mode in ['dxp', 'both']:
            return self.perturbers['dxp'](inputs_embeds)
        if self.noise_mode == 'dc':
            self._store_fx('embedding', inputs_embeds)
        return inputs_embeds

    def inject_between_blocks(self, hidden_states, i):
        to_save = hidden_states
        if self.dim_reducer and self.fl_config and self.fl_config.reducer_enable and i == self.dim_reducer.config.layer - 1:
            half = hidden_states.dtype == float16
            self.dim_reducer.to(hidden_states.device)
            to_save, hidden_states = self.dim_reducer(hidden_states)
            if half:
                hidden_states = hidden_states.half()
        if self.fl_config and self.fl_config.attack_mode:
            if i == self.fl_config.split_point_1 - 1 and self.fl_config.attack_mode == 'b2tr':
                if self.noise_mode == 'gaussian':
                    return self.perturbers['gaussian'](hidden_states), hidden_states
                return to_save, hidden_states
            elif i == self.fl_config.split_point_2 and self.fl_config.attack_mode == 'tr2t':
                return to_save, hidden_states

        if self.fl_config is not None and self.fl_config.trigger_hook and i == self.fl_config.split_point_1 - 1:  # bottom-trunk
            for hook in self.b2tr_hooks:
                hook(to_save)

        if self.fl_config is not None and self.noise_mode == 'gaussian' and i == self.fl_config.split_point_1 - 1:
            to_save = hidden_states = self.perturbers['gaussian'](to_save)
        if self.training and self.fl_config is not None and self.fl_config.collect_intermediates:
            if i == self.fl_config.split_point_1 - 1:  # bottom-trunk
                to_save.retain_grad()
                self._store_fx(i, to_save)
                for hook in self.b2tr_hooks:
                    hook(to_save)
            elif i == self.fl_config.split_point_2 - 1:  # trunk-top
                to_save.retain_grad()
                self._store_fx(i, to_save)
            elif self.fl_config.collect_all_layers:
                to_save.retain_grad()
                self._store_fx(i, to_save)
        return None, hidden_states


class SplitWrapperModel(SplitModel, ABC):
    """
    最外层的切分模型,需要实现对top-trunk-bottom三个区域参数的读取与加载

    """

    def __init__(self, llm_type='decoder-only', *args, **kwargs):
        super(SplitWrapperModel, self).__init__(*args, **kwargs)
        self.type = llm_type
        self.task_type = 'lm'

    @abstractmethod
    def get_adapter_module_regex(self):
        """
        获得要加上Adapter的Module名称的正则表达式
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_inter(self, detach=True):
        raise NotImplementedError

    @abstractmethod
    def change_noise(self, scale, mode=None):
        raise NotImplementedError

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

    def load_top_params(self, params, skip_frozen=True, skip_lora=False):
        for (nm, p), p1 in zip(self.get_top_params(), params):
            if skip_frozen and not p.requires_grad:
                continue
            if skip_lora and 'lora' in nm:
                continue
            p.data = p1.data.to(self.device)

    def load_bottom_params(self, params, skip_frozen=True, skip_lora=False):
        for (nm, p), p1 in zip(self.get_bottom_params(), params):
            if skip_frozen and not p.requires_grad:
                continue
            if skip_lora and 'lora' in nm:
                continue
            p.data = p1.data.to(self.device)

    def load_trunk_params(self, params, skip_frozen=True, skip_lora=False):
        for (nm, p), p1 in zip(self.get_trunk_params(), params):
            if skip_frozen and not p.requires_grad:
                continue
            if skip_lora and 'lora' in nm:
                continue
            p.data = p1.data.to(self.device)

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
                    if param.dtype == torch.float32:
                        param.requires_grad = True
            if not self.fl_config.use_lora_at_top:
                for name, param in res.get_top_params(trainable_only=False):
                    if param.dtype == torch.float32:
                        param.requires_grad = True
            if not self.fl_config.use_lora_at_bottom:
                for name, param in res.get_bottom_params(trainable_only=False):
                    if param.dtype == torch.float32:
                        param.requires_grad = True
        self.adapter_added = True
        return res
