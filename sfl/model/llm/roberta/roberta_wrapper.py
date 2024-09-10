from typing import Optional, Union, Tuple

import regex
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from sfl.utils.args import FLConfig
from sfl.model.llm.roberta.roberta_split import RobertaSplitModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.exp import register_model


class RobertaSplitWrapper(SplitWrapperModel):

    def change_noise(self, scale, mode=None):
        self.roberta.change_noise(scale, mode)

    @staticmethod
    def _get_block_num(param_name: str):
        # 获得该参数所属的block的块号，不属于block则返回-1
        r = regex.findall('\.layer\.[0-9]+', param_name)
        return int(r[0].split('.')[-1]) if len(r) > 0 else -1

    def get_adapter_module_regex(self):
        # Trunk部分(h.start~h.end)的proj/fc/_attn模块
        if self.fl_config is not None:
            blocks = []
            if self.fl_config.use_lora_at_bottom:
                blocks += [str(i) for i in range(self.fl_config.split_point_1)]
            if self.fl_config.use_lora_at_trunk:
                blocks += [str(i) for i in range(self.fl_config.split_point_1, self.fl_config.split_point_2)]
            if self.fl_config.use_lora_at_top:
                blocks += [str(i) for i in range(self.fl_config.split_point_2, self.config.num_hidden_layers)]
            reg = rf".*\.layer\.({'|'.join(blocks)})\..*(.+query|key|value|dense)$"
            return reg
        return ""

    def get_all_inter(self, detach=True):
        return self.roberta.get_all_inter(detach)

    def config_sfl(self, config: FLConfig, *args, **kwargs):
        super(RobertaSplitWrapper, self).config_sfl(config, *args, **kwargs)
        self.roberta.config_sfl(config, *args, **kwargs)


@register_model('roberta', register_for_prefix=True, dir_names='FacebookAI/$model_name/')
class RobertaForSequenceClassificationSplitModel(RobertaForSequenceClassification, RobertaSplitWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaSplitModel(config, add_pooling_layer=False)
        self.task_type = 'classification'

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        """
        SFL: 打断前传
        """
        if self.fl_config and self.fl_config.attack_mode:
            return outputs

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
