import argparse
from abc import ABC
from typing import Any

from tokenizers import Tokenizer

from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator


class Attacker(ABC):

    def __init__(self, name=None, arg_clz=None):
        self.name = name
        self.arg_clz = arg_clz

    def load_attacker(self, args, aargs, llm: SplitWrapperModel = None):
        raise NotImplementedError

    def attack(self, args, aargs,
               llm: SplitWrapperModel, tokenizer: Tokenizer, simulator: SFLSimulator, batch, b2tr_inter, tr2t_inter,
               all_inters, init=None) -> dict[str, Any]:
        raise NotImplementedError

    def parse_arguments(self, args, prefix: str):
        kwargs = {}
        for k, v in vars(args).items():
            if k.startswith(prefix):
                inner_name = k[len(prefix) + 1:]
                if inner_name == 'enable':
                    continue
                kwargs[inner_name] = v
        return self.arg_clz(**kwargs)
