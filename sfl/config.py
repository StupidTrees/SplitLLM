from dataclasses import dataclass

from transformers import PretrainedConfig

data_root = '/data/stupidtree/data'

dataset_cache_dir = f'{data_root}/sfl/datasets/'
model_download_dir = f'{data_root}/sfl/models/'
model_cache_dir = f'{data_root}/sfl/cache/'
attacker_path = f'{data_root}/sfl/models/attacker/'
mapper_path = f'{data_root}/sfl/models/mapper/'
reducer_path = f'{data_root}/sfl/models/reducer/'
fsha_path = f'{data_root}/sfl/models/attacker-fsha/'

DRA_train_label = {
    'codealpaca': 'test',
    'dialogsum': 'validation',
    'piqa': 'validation',
    'piqa-mini': 'validation',
    'gsm8k': 'test',
    'wikitext': 'validation',
    'sensireplaced': 'validation',
    'sensimarked': 'validation',
    'sensimasked': 'validation',
    'imdb': 'unsupervised',
    'hc3cn': 'baike',
    'imagewoof': 'validation',
}

DRA_test_label = {nm: 'test' for nm in DRA_train_label.keys()}
DRA_test_label['hc3cn'] = 'finance'
DRA_test_label['imagewoof'] = 'validation'

dxp_moe_range = {0.08, 0.21, 0.38}
gaussian_moe_range = {3.0, 5.0, 8.0}
dc_moe_range = {12.0, 24.0, 48.0}


@dataclass
class FLConfig:
    global_round: int = 0
    client_epoch: int = 3
    client_steps: int = 50
    max_global_step: int = -1  # 最多进行的global steo
    client_evaluate_freq: int = 10  # 几个Step上报一次
    client_per_round: float = 1.0
    split_point_1: int = 2
    split_point_2: int = 10
    use_lora_at_trunk: bool = True
    use_lora_at_bottom: bool = False
    use_lora_at_top: bool = False
    use_lora_at_embed: bool = False
    collect_intermediates: bool = True  # 是否记录中间结果
    collect_all_layers: bool = False  # 是否记录所有层的中间结果
    trigger_hook: bool = False
    top_and_bottom_from_scratch: str = 'False'  # 设置为True，Client将不采用预训练的Top和Bottom参数
    attack_mode: str | None = None  # 'b2tr' or 'tr2b' or 'self' or None
    noise_mode: str = 'none'
    noise_scale_dxp: float = 0.0
    noise_scale_grad: float = 0.0
    noise_scale_gaussian: float = 0.0
    noise_beta_dc: float = 0.1
    dataset_type: str = 'train'
    batch_size: int = 2
    reducer_enable: bool = False
    lr: float = 2e-5


@dataclass
class SIPAttackerArguments:
    enable: bool = True
    path: str = attacker_path
    b2tr_enable: bool = True
    b2tr_layer: int = -1
    b2tr_target_layer: int = -1
    tr2t_enable: bool = True
    tr2t_layer: int = -1
    tr2t_target_layer: int = -1
    model: str = 'gru'  # DRAttacker Model
    dataset: str = None  # what dataset the DRAttacker is trained on
    # train_label: str = 'validation'  # training dataset of that model
    train_frac: float = 1.0  # percentage of dataset used for training
    prefix: str = 'normal'
    target_model_name: str = None
    target_dataset: str = None
    target_system_sps: str = None
    target_model_load_bits: int = -1
    larger_better: bool = True
    attack_all_layers: bool = False


@dataclass
class MapperConfig:
    path: str = mapper_path
    from_layer: int = 6
    to_layer: int = 1
    dataset: str = None  # what dataset the DRAttacker is trained on
    train_frac: float = 1.0  # percentage of dataset used for training
    prefix: str = 'normal'
    target_model_name: str = None
    target_dataset: str = None
    target_model_load_bits: int = -1
    larger_better: bool = False
