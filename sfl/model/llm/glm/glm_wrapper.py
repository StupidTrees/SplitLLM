from typing import Optional, Tuple

import torch
from regex import regex
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from sfl.utils.args import FLConfig
from sfl.model.llm.glm.configuration_chatglm import ChatGLMConfig
from sfl.model.llm.glm.glm_split import ChatGLMSplitModel
from sfl.model.llm.glm.modeling_chatglm import ChatGLMForConditionalGeneration
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.exp import register_model


@register_model('chatglm',requiring_quantization=True,dir_names='THUDM/chatglm3-6b')
class ChatGLMForConditionalGenerationSplit(ChatGLMForConditionalGeneration, SplitWrapperModel):

    def config_sfl(self, config: FLConfig, *args, **kwargs):
        super(ChatGLMForConditionalGenerationSplit, self).config_sfl(config, *args, **kwargs)
        self.transformer.config_sfl(config, *args, **kwargs)

    def get_adapter_module_regex(self):
        if self.fl_config is not None:
            blocks = []
            if self.fl_config.use_lora_at_bottom:
                blocks += [str(i) for i in range(self.fl_config.split_point_1)]
            if self.fl_config.use_lora_at_trunk:
                blocks += [str(i) for i in range(self.fl_config.split_point_1, self.fl_config.split_point_2)]
            if self.fl_config.use_lora_at_top:
                blocks += [str(i) for i in range(self.fl_config.split_point_2, self.config.num_layers)]
            reg = rf".*\.layers\.({'|'.join(blocks)})\..*(.+self_attention|dense)$"
            return reg
        return ""

    @staticmethod
    def _get_block_num(param_name: str):
        # 获得该参数所属的block的块号，不属于block则返回-1
        r = regex.findall('\.layers\.[0-9]+', param_name)
        return int(r[0].split('.')[-1]) if len(r) > 0 else -1

    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)
        self.max_sequence_length = config.max_length
        self.transformer = ChatGLMSplitModel(config, empty_init=empty_init, device=device)
        self.config = config
        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def get_all_inter(self, detach=True):
        return self.transformer.get_all_inter(detach=detach)

    def change_noise(self, scale, mode=None):
        super(ChatGLMForConditionalGenerationSplit, self).change_noise(scale, mode)
        self.transformer.change_noise(scale, mode)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.fl_config and self.fl_config.attack_mode:
            return transformer_outputs

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
