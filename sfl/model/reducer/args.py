import dataclasses

from transformers import PretrainedConfig

from sfl.config import reducer_path


@dataclasses.dataclass
class ReducerArgument:
    dataset: str = None
    target_model: str = None
    target_model_load_bits: int = -1
    train_label: str = 'train'
    train_frac: float = 1.0
    layer: int = 6
    alpha: int = 128
    path: str = reducer_path


@dataclasses.dataclass
class ReductionTrainingArguments:
    epochs: int = 15
    batch_size: int = 6
    lr: float = 1e-3
    opt: str = 'adam'
    test_frac: float = 0.1
    checkpoint_freq: int = 2
    save_checkpoint: bool = True


class DRConfig(PretrainedConfig):
    n_embed: int = 0
    alpha: int = 8
    layer: int = 6  # layer=6 means the output of block #5 will be transformed

