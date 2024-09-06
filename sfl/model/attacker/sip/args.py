import dataclasses

from sfl.config import attacker_path


@dataclasses.dataclass
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
    dataset: str = None  # what data the DRAttacker is trained on
    # train_label: str = 'validation'  # training data of that model
    train_frac: float = 1.0  # percentage of data used for training
    prefix: str = 'normal'
    target_model_name: str = None
    target_dataset: str = None
    target_system_sps: str = None
    target_model_load_bits: int = -1
    larger_better: bool = True
    attack_all_layers: bool = False


@dataclasses.dataclass
class InversionModelTrainingArgument:
    test_frac:float = 0.1
    optim:str = 'adam'
    lr:float = 1e-3
    weight_decay:float = 1e-5
    epochs:int = 20
    gating_epochs:int = 15
    ft_epochs:int = 4
    batch_size:int = 6
    log_to_wandb:bool = False
    save_checkpoint:bool = True
    checkpoint_freq:int = 5
    save_threshold:float = 0.1