from dataclasses import dataclass

data_root = '/data/stupidtree/data'

dataset_cache_dir = f'{data_root}/sfl/datasets/'
model_download_dir = f'{data_root}/sfl/models/'
model_cache_dir = f'{data_root}/sfl/cache/'
attacker_path = f'{data_root}/sfl/models/attacker/'
mapper_path = f'{data_root}/sfl/models/mapper/'
reducer_path = f'{data_root}/sfl/models/reducer/'
lora_path = f'{data_root}/sfl/models/lora/'
fsha_path = f'{data_root}/sfl/models/attacker-fsha/'

DRA_train_label = {
    'codealpaca': 'test',
    'dialogsum': 'validation',
    'piqa': 'validation',
    'qnli': 'validation',
    'mrpc': 'validation',
    'rte': 'validation',
    'stsb': 'validation',
    'cola': 'test',
    'piqa-mini': 'validation',
    'gsm8k': 'test',
    'wikitext': 'validation',
    'wikitext103': 'validation',
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
gaussian_clipping_threshold = 2000
dc_moe_range = {8.0, 32.0, 64.0}

@dataclass
class FLConfig:
    global_round: int = 0
    client_epoch: int = 3
    client_steps: int = 50
    max_global_step: int = -1  # maximum global step
    client_evaluate_freq: int = 10  # evaluate every n client steps
    client_per_round: float = 1.0
    split_point_1: int = 2
    split_point_2: int = 10
    split_mode: str = 'hidden'  # 'hidden' or 'qk'
    use_lora_at_trunk: bool = True
    use_lora_at_bottom: bool = False
    use_lora_at_top: bool = False
    use_lora_at_embed: bool = False
    collect_intermediates: bool = True  # record intermediate results or not
    collect_all_layers: bool = False  # record all layers or not
    trigger_hook: bool = False
    top_and_bottom_from_scratch: str = 'False'  # 'True' or 'False', whether to train top and bottom from scratch
    attack_mode: str | None = None  # 'b2tr' or 'tr2b' or 'self' or None
    noise_mode: str = 'none'
    noise_scale: float = 0.0
    dataset_type: str = 'train'
    batch_size: int = 2
    reducer_enable: bool = False
    lr: float = 2e-5
