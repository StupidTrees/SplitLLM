import logging
from typing import Optional, Union, Tuple, List

import torch
from regex import regex
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from sfl.model.split_model import SplitModel
from sfl.simulator.param_keeper import ParameterKeeper
from sfl.utils import FLConfig

logger = logging.getLogger(__name__)


class LLAMA2SplitLMHeadModel(LlamaForCausalLM, SplitModel):
    """
    最后一层用于文本生成
    """

    def __init__(self, config):
        super(LLAMA2SplitLMHeadModel, self).__init__(config)
        self.model = LLAMA2SplitModel(config)

    def get_bottom_to_trunk_fx(self):
        return self.model.get_bottom_to_trunk_fx()

    def get_trunk_to_top_fx(self):
        return self.model.get_trunk_to_top_fx()

    def get_top_to_trunk_grad(self):
        return self.model.get_top_to_trunk_grad()

    def get_trunk_to_bottom_grad(self):
        return self.model.get_trunk_to_bottom_grad()

    def get_trunk_adapter_module_regex(self):
        # Trunk部分(h.start~h.end)的proj/fc/_attn模块
        if self.fl_config is not None:
            reg = rf".*\.layer\.({'|'.join([str(i) for i in range(self.fl_config.split_point_1, self.fl_config.split_point_2)])})\..*(.+attn|proj|fc)$"
            return reg
        return ""

    @staticmethod
    def _get_block_num(param_name: str):
        # 获得该参数所属的block的块号，不属于block则返回-1
        r = regex.findall('\.h\.[0-9]+', param_name)
        return int(r[0].split('.')[-1]) if len(r) > 0 else -1

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

    def config_sfl(self, config: FLConfig, param_keeper: ParameterKeeper|None):
        super(LLAMA2SplitLMHeadModel, self).config_sfl(config, param_keeper)
        self.model.config_sfl(config, param_keeper)

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
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

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
        
        # @ add for fl
        if self.fl_config and self.fl_config.attack_mode:
            return outputs

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
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



class LLAMA2SplitModel(LlamaModel, SplitModel):
    """
    主模型，主要在FP过程中收集中间输出和梯度
    """

    def get_bottom_params(self, trainable_only=True):
        pass

    def get_top_params(self, trainable_only=True):
        pass

    def get_trunk_params(self, trainable_only=True):
        pass

    def get_trunk_adapter_module_regex(self):
        pass

    def get_bottom_to_trunk_fx(self):
        if 'trunk_to_top' in self.intermediate_fx:
            return self.intermediate_fx['bottom_to_trunk'].detach().cpu()
        return []

    def get_trunk_to_top_fx(self):
        if 'bottom_to_trunk' in self.intermediate_fx:
            return self.intermediate_fx['trunk_to_top'].detach().cpu()
        return []

    def get_top_to_trunk_grad(self):
        if 'trunk_to_top' in self.intermediate_fx:
            return self.intermediate_fx['trunk_to_top'].grad.clone().detach().cpu()
        return []

    def get_trunk_to_bottom_grad(self):
        if 'bottom_to_trunk' in self.intermediate_fx:
            return self.intermediate_fx['bottom_to_trunk'].grad.clone().detach().cpu()
        return []

    def _store_bottom_to_trunk_fx(self, fx):
        self.intermediate_fx['bottom_to_trunk'] = fx

    def _store_trunk_to_top_fx(self, fx):
        self.intermediate_fx['trunk_to_top'] = fx

    def __init__(self, config):
        super().__init__(config)
        self.intermediate_fx = {}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # @ add for fl
            # SFL: store intermediate hidden states
            if self.fl_config and self.fl_config.attack_mode:
                if idx == self.fl_config.split_point_1 - 1 and self.fl_config.attack_mode == 'b2tr':
                    return hidden_states
                elif idx == self.fl_config.split_point_2 and self.fl_config.attack_mode == 'tr2t':
                    return hidden_states

            if self.training and self.fl_config is not None and self.fl_config.collect_intermediates:
                if idx == self.fl_config.split_point_1 - 1:  # bottom-trunk
                    hidden_states.retain_grad()
                    self._store_bottom_to_trunk_fx(hidden_states)
                elif idx == self.fl_config.split_point_2:  # trunk-top
                    hidden_states.retain_grad()
                    self._store_trunk_to_top_fx(hidden_states)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )