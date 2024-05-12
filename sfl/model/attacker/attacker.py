from abc import ABC


class Attacker(ABC):

    def attack(self, b2tr_inter, tr2t_inter, all_inters, init=None):
        raise NotImplementedError

