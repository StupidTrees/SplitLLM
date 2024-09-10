from abc import ABC, abstractmethod

from tokenizers import Tokenizer

from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator
from sfl.utils.args import PrefixArgumentParser


class Attacker(ABC):
    """
    Base class for all attackers
    """
    arg_clz = None

    @abstractmethod
    def load_attacker(self, args, aargs, llm: SplitWrapperModel = None):
        raise NotImplementedError

    @abstractmethod
    def attack(self, args, aargs: arg_clz,
               llm: SplitWrapperModel, tokenizer: Tokenizer, simulator: SFLSimulator, batch, b2tr_inter, tr2t_inter,
               all_inters, init=None):
        raise NotImplementedError

    def parse_arguments(self, args, prefix: str):
        parser = PrefixArgumentParser([self.__class__.arg_clz], prefix=prefix)
        return parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
