from typing import Optional, Union

import regex
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import ViTForImageClassification, ViTConfig
from transformers.modeling_outputs import ImageClassifierOutput

from sfl.utils.args import FLConfig
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.model.llm.vit.vit_split import ViTSplit
from sfl.simulator.param_keeper import ParameterKeeper


class ViTForImageClassificationSplit(ViTForImageClassification, SplitWrapperModel):

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTSplit(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()
        self.task_type = 'classification'

    @staticmethod
    def _get_block_num(param_name: str):
        # 获得该参数所属的block的块号，不属于block则返回-1
        r = regex.findall('\.layer\.[0-9]+', param_name)
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
        reg = rf".*\.layer\.({'|'.join(blocks)})\..*(.+v_proj|q_proj)$"
        return reg

    def get_all_inter(self, detach=True):
        return self.vit.get_all_inter(detach)

    def change_noise(self, scale, mode=None):
        super(ViTForImageClassificationSplit, self).change_noise(scale, mode)
        self.vit.change_noise(scale, mode)

    def config_sfl(self, config: FLConfig, param_keeper: ParameterKeeper | None, b2tr_hooks: list = None):
        super(ViTForImageClassificationSplit, self).config_sfl(config, param_keeper, b2tr_hooks)
        self.vit.config_sfl(config, param_keeper, b2tr_hooks)

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        if self.fl_config and self.fl_config.attack_mode:
            return outputs

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
