from typing import Optional, Union, Tuple

import torch
from transformers import MambaCache
from transformers.models.mamba.modeling_mamba import MambaOutput, MambaModel

from sfl.model.llm.split_model import SplitModel


class MambaSplitModel(MambaModel, SplitModel):

    def __init__(self, config):
        super().__init__(config)
        self.intermediate_fx = {}

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            cache_params: Optional[MambaCache] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = MambaCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        """
        SFL: embedding 后插入噪声
        """
        inputs_embeds = self.inject_after_embedding(inputs_embeds)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        for idx, mixer_block in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position
                )
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params, cache_position=cache_position)

            """
            SFL: 在每个 block 后插入噪声
            """
            interrupt, hidden_states = self.inject_between_blocks(hidden_states, idx)
            if interrupt is not None:
                return interrupt

            if output_hidden_states:
                all_hidden_states += hidden_states,

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += hidden_states,

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )
