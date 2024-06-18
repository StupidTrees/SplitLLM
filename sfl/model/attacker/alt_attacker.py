from abc import ABC
from dataclasses import dataclass

from tokenizers import Tokenizer

from sfl.config import FLConfig
from sfl.model.attacker.base import Attacker
from sfl.model.attacker.dlg_attacker import TAGAttacker, TAGArguments
from sfl.model.attacker.eia_attacker import EmbeddingInversionAttacker, EIAArguments
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator


@dataclass
class ALTArguments:
    enable: bool = False
    epochs: int = 2
    b_epochs: int = 300
    b_beta: float = 0.85
    b_lr: float = 0.09
    b_init_temp: float = 1.0
    b_softmax = True
    f_at: str = 'b2tr'
    f_mapped_to: int = 1
    f_epochs: int = 72000
    f_lr: float = 0.09
    f_wd: float = 0.01
    f_temp: float = 0.2


class ALTAttacker(Attacker, ABC):
    """
    Attacker that combines TAG and EIA attacks
    """
    arg_clz = ALTArguments

    def __init__(self, fl_config: FLConfig, model: SplitWrapperModel):
        super().__init__()
        self.tag_attacker = TAGAttacker(fl_config, model)
        self.eia_attacker = EmbeddingInversionAttacker()

    def _get_b_f_args(self, aargs: ALTArguments):
        b_arg = TAGArguments(epochs=aargs.b_epochs, beta=aargs.b_beta, lr=aargs.b_lr, init_temp=aargs.b_init_temp,
                             softmax=aargs.b_softmax)
        f_arg = EIAArguments(at=aargs.f_at, mapped_to=aargs.f_mapped_to, epochs=aargs.f_epochs, lr=aargs.f_lr,
                             wd=aargs.f_wd, temp=aargs.f_temp)
        return b_arg, f_arg

    def load_attacker(self, args, aargs: arg_clz, llm: SplitWrapperModel = None, tokenizer: Tokenizer = None):
        b_arg, f_arg = self._get_b_f_args(aargs)
        self.tag_attacker.load_attacker(args, b_arg, llm, tokenizer)
        self.eia_attacker.load_attacker(args, f_arg, llm, tokenizer)

    def attack(self, args, aargs: arg_clz,
               llm: SplitWrapperModel, tokenizer: Tokenizer, simulator: SFLSimulator, batch, b2tr_inter, tr2t_inter,
               all_inters, init=None):
        atk_init = init
        b_arg, f_arg = self._get_b_f_args(aargs)
        for round in range(aargs.epochs):
            f_res = self.eia_attacker.attack(args, f_arg, llm, tokenizer, simulator, batch, b2tr_inter,
                                             tr2t_inter, all_inters, atk_init)
            b_res = self.tag_attacker.attack(args, b_arg, llm, tokenizer, simulator, batch, b2tr_inter,
                                             tr2t_inter, all_inters, f_res)
            atk_init = b_res
        return {'f': f_res, 'b': b_res}
