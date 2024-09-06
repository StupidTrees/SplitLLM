from dataclasses import dataclass
from sfl.config import mapper_path


@dataclass
class EIAArguments:
    enable: bool = False
    at: str = 'b2tr'
    mapper_target_model_name: str = None
    mapper_target_model_load_bits: int = -1
    mapper_train_frac: float = 1.0
    mapper_path: str = mapper_path
    mapper_dataset: str = ''
    mapper_targets: str = None
    mapped_to: int = -1
    epochs: int = 72000
    lr: float = 0.09
    wd: float = 0.01
    temp: float = 0.2
    cross_model: str = None


@dataclass
class MapperTrainingArguments:
    batch_size: int = 4
    test_frac: float = 0.1
    lr: float = 1e-3
    wd: float = 0.01
    checkpoint_freq: int = 2
    epochs: int = 15
    opt: str = 'adam'
    log_to_wandb: bool = False
