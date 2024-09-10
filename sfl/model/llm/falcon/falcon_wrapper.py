from typing import Optional, Tuple, Union

import regex
import torch
from torch.nn import CrossEntropyLoss
from transformers import FalconForCausalLM, FalconConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from sfl.utils.args import FLConfig
from sfl.model.llm.falcon.falcon_split import FalconSplitModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.exp import register_model


@register_model('falcon', requiring_quantization=True, dir_names='tiiuae/falcon-7b-instruct')
class FalconForCausalLMSplit(FalconForCausalLM, SplitWrapperModel):

    def config_sfl(self, config: FLConfig, *args, **kwargs):
        super(FalconForCausalLMSplit, self).config_sfl(config, *args, **kwargs)
        self.transformer.config_sfl(config, *args, **kwargs)

    @staticmethod
    def _get_block_num(param_name: str):
        # 获得该参数所属的block的块号，不属于block则返回-1
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
            blocks += [str(i) for i in range(self.fl_config.split_point_2, self.config.num_hidden_layers)]
        reg = rf".*\.h\.({'|'.join(blocks)})\..*(.+self_attention|dense)$"
        if self.fl_config.use_lora_at_embed:
            reg = rf"^({reg}|.*(.+wte|wpe|embeddings).*)$"
        if self.fl_config.use_lora_at_top:
            reg = rf"^({reg}|.*(.+lm_head).*)$"
        return reg

    def get_all_inter(self, detach=True):
        return self.transformer.get_all_inter(detach)

    def change_noise(self, scale, mode=None):
        super(FalconForCausalLMSplit, self).change_noise(scale, mode)
        self.transformer.change_noise(scale, mode)

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.transformer = FalconSplitModel(config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.fl_config and self.fl_config.attack_mode:
            return transformer_outputs

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
