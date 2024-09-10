from typing import Optional, List, Tuple, Union

import regex
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from sfl.utils.args import FLConfig
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.wizard.wizard_split import WizardSplitModel
from sfl.simulator.param_keeper import ParameterKeeper
from sfl.utils.exp import register_model


@register_model('wizard', requiring_quantization=True, dir_names='lucyknada/microsoft_WizardLM-2-7B')
class WizardForCausalLMSplit(MistralForCausalLM, SplitWrapperModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = WizardSplitModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def config_sfl(self, config: FLConfig, param_keeper: ParameterKeeper | None = None, b2tr_hooks: list = None, *args,
                   **kwargs):
        super(WizardForCausalLMSplit, self).config_sfl(config, param_keeper, b2tr_hooks, *args, **kwargs)
        self.model.config_sfl(config, param_keeper, b2tr_hooks, *args, **kwargs)

    @staticmethod
    def _get_block_num(param_name: str):
        # 获得该参数所属的block的块号，不属于block则返回-1
        r = regex.findall('\.layers\.[0-9]+', param_name)
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
        reg = rf".*\.layers\.({'|'.join(blocks)})\..*(.+v_proj|q_proj)$"
        if self.fl_config.use_lora_at_embed:
            reg = rf"^({reg}|.*embed_tokens.*)$"
        return reg

    def get_all_inter(self, detach=True):
        return self.model.get_all_inter(detach)

    def change_noise(self, scale, mode=None):
        super(WizardForCausalLMSplit, self).change_noise(scale, mode)
        self.model.change_noise(scale, mode)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        """
        SFL: 打断前传
        """
        if self.fl_config and self.fl_config.attack_mode:
            return outputs

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
