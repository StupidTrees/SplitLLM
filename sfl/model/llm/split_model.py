from abc import ABC, abstractmethod

import regex
import torch
from peft import LoraConfig, get_peft_model
from torch import nn, float16

from sfl.utils.args import FLConfig
from sfl.model.noise.dxp import DxPrivacy
from sfl.model.noise.fdp import GaussianPerturber
from sfl.model.reducer.reducer_models import DimReduction
from sfl.simulator.param_keeper import ParameterKeeper
from sfl.utils.model import Intermediate, get_embedding_layer


class SplitModel(nn.Module, ABC):
    """
    The model simulated for three-part split SFL (private-label)
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
        self.perturbers = {}
        self.inner_loop = False  # set true when calling forward during a forward

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
        if config.noise_mode == 'dxp':
            self.perturbers['dxp'] = DxPrivacy(get_embedding_layer(self), self.config.vocab_size,
                                               config.noise_scale)
        elif config.noise_mode == 'gaussian':
            self.perturbers['gaussian'] = GaussianPerturber(config.noise_scale)

        self.change_noise(config.noise_scale, config.noise_mode)
        # More perturbation to be added here...

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
        if self.inner_loop:
            return inputs_embeds
        if self.noise_mode in ['dxp', 'both']:
            return self.perturbers['dxp'](inputs_embeds)
        if self.noise_mode in ['dc']:
            self._store_fx('embedding', inputs_embeds)
        return inputs_embeds

    def inject_between_blocks(self, hidden_states, i):
        if self.inner_loop:
            if i == self.fl_config.split_point_1 - 1 and self.fl_config.attack_mode == 'b2tr':
                return hidden_states, hidden_states
            elif i == self.fl_config.split_point_2 and self.fl_config.attack_mode == 'tr2t':
                return hidden_states, hidden_states
            return None, hidden_states

        to_save = hidden_states
        if self.dim_reducer and self.fl_config and self.fl_config.reducer_enable and i == self.dim_reducer.config.layer - 1:
            half = hidden_states.dtype == float16
            self.dim_reducer.to(hidden_states.device)
            to_save, hidden_states = self.dim_reducer(hidden_states)
            if half:
                hidden_states = hidden_states.half()
        if self.fl_config and self.fl_config.attack_mode:
            if i == self.fl_config.split_point_1 - 1 and self.fl_config.attack_mode == 'b2tr':
                if self.noise_mode in ['gaussian', 'dc-sim']:
                    return self.perturbers[self.noise_mode](hidden_states), hidden_states
                return to_save, hidden_states
            elif i == self.fl_config.split_point_2 and self.fl_config.attack_mode == 'tr2t':
                return to_save, hidden_states
        if self.fl_config is not None and self.fl_config.trigger_hook and i == self.fl_config.split_point_1 - 1:  # bottom-trunk
            for hook in self.b2tr_hooks:
                hook(to_save)

        if self.fl_config is not None and self.noise_mode in ['gaussian',
                                                              'dc-sim'] and i == self.fl_config.split_point_1 - 1:
            to_save = hidden_states = self.perturbers[self.noise_mode](to_save)
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
    Outer wrapper for SplitModel

    """

    def __init__(self, llm_type='decoder-only', *args, **kwargs):
        super(SplitWrapperModel, self).__init__(*args, **kwargs)
        self.type = llm_type
        self.task_type = 'lm'

    @abstractmethod
    def get_adapter_module_regex(self):
        """
        Get the regex pattern for modules that need to be adapted by LoRA
        :return: str
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
        if self.adapter_added:
            return self
        if (not self.fl_config.use_lora_at_top) and (not self.fl_config.use_lora_at_bottom) and (
                not self.fl_config.use_lora_at_trunk):
            return self
        lora_config = LoraConfig(target_modules=self.get_adapter_module_regex())
        res = get_peft_model(self, lora_config)
        if restore_rest:
            # PEFT will freeze all model parameters, need to restore the rest
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
