from typing import Optional, Tuple, Union

import torch
from regex import regex
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from sfl.utils.args import FLConfig
from sfl.model.llm.bert.bert_split import BertSplitModel
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.utils.exp import register_model


class BertSplitWrapper(SplitWrapperModel):
    """
    最外层模型，需要重写get_all_inter(), config_sfl()两个方法
    """

    @staticmethod
    def _get_block_num(param_name: str):
        r = regex.findall('\.layer\.[0-9]+', param_name)
        return int(r[0].split('.')[-1]) if len(r) > 0 else -1

    def get_adapter_module_regex(self):
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
        return self.bert.get_all_inter(detach)

    def config_sfl(self, config: FLConfig, *args, **kwargs):
        super(BertSplitWrapper, self).config_sfl(config, *args, **kwargs)
        self.bert.config_sfl(config, *args, **kwargs)

    def change_noise(self, noise_scale, noise_mode=None):
        self.bert.change_noise(noise_scale, noise_mode)


@register_model('bert', register_for_prefix=True, dir_names='google-bert/$model_name-uncased/')
class BertForSequenceClassificationSplitModel(BertForSequenceClassification, BertSplitWrapper):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertSplitModel(config)
        self.task_type = 'classification'

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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
        SL: Interrupt the forward process
        """
        if self.fl_config and self.fl_config.attack_mode:
            return outputs

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
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
