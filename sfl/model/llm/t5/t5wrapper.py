import copy
import warnings
from typing import Optional, Tuple, Union

import torch
from regex import regex
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from sfl.utils.args import FLConfig
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.t5.t5split import T5SplitStack
from sfl.utils.exp import register_model


class T5SplitWrapper(SplitWrapperModel):

    def __init__(self):
        super(T5SplitWrapper, self).__init__(llm_type='encoder-decoder')

    @staticmethod
    def _get_block_num(param_name: str):
        # 获得该参数所属的block的块号，不属于block则返回-1
        r = regex.findall('\.block\.[0-9]+', param_name)
        return int(r[0].split('.')[-1]) if len(r) > 0 else -1

    def get_bottom_params(self, trainable_only=True):
        for nm, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if 'decoder' in nm and self._get_block_num(nm) >= self.fl_config.split_point_1:
                break
            else:
                yield nm, p

    def get_top_params(self, trainable_only=True):
        trunk = False
        for nm, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if 'decoder' in nm and self._get_block_num(nm) >= self.fl_config.split_point_2:
                trunk = True
            if trunk:
                yield nm, p

    def get_trunk_params(self, trainable_only=True):
        for nm, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if 'decoder' in nm and self.fl_config.split_point_1 <= self._get_block_num(
                    nm) < self.fl_config.split_point_2:
                yield nm, p

    def get_adapter_module_regex(self):
        # Trunk部分(h.start~h.end)的proj/fc/_attn模块
        if self.fl_config is not None:
            blocks = []
            if self.fl_config.use_lora_at_bottom:
                blocks += [str(i) for i in range(self.fl_config.split_point_1)]
            if self.fl_config.use_lora_at_trunk:
                blocks += [str(i) for i in range(self.fl_config.split_point_1, self.fl_config.split_point_2)]
            if self.fl_config.use_lora_at_top:
                blocks += [str(i) for i in range(self.fl_config.split_point_2, self.config.num_layers)]
            reg = rf"decoder\.block\.({'|'.join(blocks)})\..*(.+q|k|v|o|wi_0|wi_1|wo)$"
            if self.fl_config.use_lora_at_bottom:
                eb = [str(i) for i in range(self.config.num_layers)]
                reg = rf"^({reg}|encoder\.block\.({'|'.join(eb)})\..*(.+q|k|v|o|wi_0|wi_1|wo)$)"
            return reg

        return ""

    def get_all_inter(self, detach=True):
        return self.decoder.get_all_inter(detach)

    def config_sfl(self, config: FLConfig, *args, **kwargs):
        super(T5SplitWrapper, self).config_sfl(config, *args, **kwargs)
        self.encoder.config_sfl(config, *args, **kwargs)
        self.decoder.config_sfl(config, *args, **kwargs)

    def change_noise(self, scale, mode=None):
        self.encoder.change_noise(scale, mode)
        self.decoder.change_noise(scale, mode)


@register_model(['t5', 'ul2'], register_for_prefix=True, dir_names=['google/$model_name', 'google/$model_name'])
class T5ForConditionalGenerationSplitModel(T5ForConditionalGeneration, T5SplitWrapper):

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.encoder = T5SplitStack(copy.deepcopy(self.encoder.config), self.shared)
        self.decoder = T5SplitStack(copy.deepcopy(self.decoder.config), self.shared)
        self.decoder.num_encoder_layers = config.num_layers

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn("?", FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        """
        SFL: 打断前传
        """
        if self.fl_config and self.fl_config.attack_mode:
            return hidden_states, decoder_outputs

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
