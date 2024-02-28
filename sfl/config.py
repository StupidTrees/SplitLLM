from dataclasses import dataclass
from typing import Any

from transformers import PretrainedConfig

dataset_cache_dir = '/root/autodl-tmp/sfl/datasets/'
model_download_dir = '/root/autodl-tmp/sfl/models/'
model_cache_dir = '/root/autodl-tmp/sfl/cache/'
attacker_path = '/root/autodl-tmp/sfl/models/attacker/'


@dataclass
class FLConfig:
    global_round: int = 0
    client_epoch: int = 3
    client_steps: int = 50
    client_evaluate_freq: int = 10  # 几个Step上报一次
    client_per_round: float = 1.0
    split_point_1: int = 2
    split_point_2: int = 10
    use_lora_at_trunk: bool = True
    use_lora_at_bottom: bool = False
    use_lora_at_top: bool = False
    collect_intermediates: bool = True  # 是否记录中间结果
    collect_all_layers: bool = False  # 是否记录所有层的中间结果
    trigger_hook: bool = False
    top_and_bottom_from_scratch: str = 'False'  # 设置为True，Client将不采用预训练的Top和Bottom参数
    attack_mode: str | None = None  # 'b2tr' or 'tr2b' or 'self' or None
    noise_mode: str = 'none'
    noise_scale: float = 0.0
    dataset_type: str = 'train'
    batch_size: int = 2


@dataclass
class DRAttackerConfig(PretrainedConfig):
    model_name: str = None
    target_model: str = None
    vocab_size: int = 0
    n_embed: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class DRAConfig:
    path: str = attacker_path
    b2tr_enable: bool = True
    b2tr_sp: int = -1
    tr2t_enable: bool = True
    tr2t_sp: int = -1
    model: str = 'gru'  # DRAttacker Model
    dataset: str = None  # what dataset the DRAttacker is trained on
    train_label: str = 'validation'  # training dataset of that model
    train_frac: float = 1.0  # percentage of dataset used for training
    prefix: str = 'normal'
    target_model_name: str = None
    target_dataset: str = None
    target_sps: str = None
