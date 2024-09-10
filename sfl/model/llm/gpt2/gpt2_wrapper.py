import logging
from typing import Optional, Tuple, Union

import regex
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, SequenceClassifierOutputWithPast

from sfl.utils.args import FLConfig
from sfl.model.llm.gpt2.gpt2_split import GPT2SplitModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.param_keeper import ParameterKeeper
from sfl.utils.exp import register_model

logger = logging.getLogger(__name__)


class GPT2SplitWrapper(SplitWrapperModel):

    def change_noise(self, scale, mode=None):
        self.transformer.change_noise(scale, mode)

    @staticmethod
    def _get_block_num(param_name: str):
        r = regex.findall('\.h\.[0-9]+', param_name)
        return int(r[0].split('.')[-1]) if len(r) > 0 else -1

    def get_adapter_module_regex(self):
        if self.fl_config is None:
            return ""
        blocks = []
        if self.fl_config.use_lora_at_bottom:
            blocks += [str(i) for i in range(self.fl_config.split_point_1)]
        if self.fl_config.use_lora_at_trunk:
            blocks += [str(i) for i in range(self.fl_config.split_point_1, self.fl_config.split_point_2)]
        if self.fl_config.use_lora_at_top:
            blocks += [str(i) for i in range(self.fl_config.split_point_2, self.config.n_layer)]
        reg = rf".*\.h\.({'|'.join(blocks)})\..*(.+attn|proj|fc)$"
        if self.fl_config.use_lora_at_embed:
            reg = rf"^({reg}|.*(.+wte|wpe).*)$"
        return reg

    def config_sfl(self, config: FLConfig, *args, **kwargs):
        super(GPT2SplitWrapper, self).config_sfl(config, *args, **kwargs)
        self.transformer.config_sfl(config, *args, **kwargs)

    def get_all_inter(self, detach=True):
        return self.transformer.get_all_inter(detach)


@register_model('gpt2', dir_names='$model_name')
class GPT2SplitLMHeadModel(GPT2LMHeadModel, GPT2SplitWrapper):

    def __init__(self, config):
        super(GPT2SplitLMHeadModel, self).__init__(config)
        self.transformer = GPT2SplitModel(config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.fl_config and self.fl_config.attack_mode:
            return transformer_outputs
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
