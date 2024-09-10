from typing import Optional, Tuple, Union

import regex
import torch
from torch.nn import CrossEntropyLoss
from transformers import MambaForCausalLM, MambaCache
from transformers.models.mamba.modeling_mamba import MambaCausalLMOutput

from sfl.utils.args import FLConfig
from sfl.model.llm.mamba.mamba_split import MambaSplitModel
from sfl.model.llm.split_model import SplitWrapperModel


class MambaSplitLMHeadModel(MambaForCausalLM, SplitWrapperModel):
    """
    Split Model for Mamba LM
    """

    def __init__(self, config):
        super(MambaSplitLMHeadModel, self).__init__(config)
        self.model = MambaSplitModel(config)

    def change_noise(self, scale, mode=None):
        self.model.change_noise(scale, mode)

    def get_adapter_module_regex(self):
        """
        target_modules = ['x_proj', 'embeddings', 'in_proj', 'out_proj']
        """
        if self.fl_config is None:
            return ""
        blocks = []
        if self.fl_config.use_lora_at_bottom:
            blocks += [str(i) for i in range(self.fl_config.split_point_1)]
        if self.fl_config.use_lora_at_trunk:
            blocks += [str(i) for i in range(self.fl_config.split_point_1, self.fl_config.split_point_2)]
        if self.fl_config.use_lora_at_top:
            blocks += [str(i) for i in range(self.fl_config.split_point_2, self.config.num_hidden_layers)]
        reg = rf".*\.layers\.({'|'.join(blocks)})\..*(.+x_proj|in_proj|out_proj)$"
        if self.fl_config.use_lora_at_embed:
            reg = rf"^({reg}|.*embeddings.*)$"
        return reg

    def get_all_inter(self, detach=True):
        return self.model.get_all_inter(detach)

    @staticmethod
    def _get_block_num(param_name: str):
        r = regex.findall('\.layers\.[0-9]+', param_name)
        return int(r[0].split('.')[-1]) if len(r) > 0 else -1

    def config_sfl(self, config: FLConfig, *args, **kwargs):
        super(MambaSplitLMHeadModel, self).config_sfl(config, *args, **kwargs)
        self.model.config_sfl(config, *args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        """
        SFL: 打断前传
        """
        if self.fl_config and self.fl_config.attack_mode:
            return mamba_outputs

        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )
