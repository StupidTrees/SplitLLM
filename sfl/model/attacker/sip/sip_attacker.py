import os

import torch
from tokenizers import Tokenizer

from sfl.model.attacker.base import Attacker
from sfl.model.attacker.sip.args import SIPAttackerArguments, InversionModelTrainingArgument
from sfl.model.attacker.sip.inversion_models import get_inverter_class
from sfl.model.attacker.sip.inversion_training import train_inversion_model, train_inversion_model_moe
from sfl.model.llm.split_model import SplitWrapperModel
from sfl.simulator.simulator import SFLSimulator, ParamRestored
from sfl.utils.args import PrefixArgumentParser
from sfl.utils.exp import required_quantization, get_dra_train_label
from sfl.utils.model import FLConfigHolder


class SIPAttacker(Attacker):
    """
    Learning-based SIP Attacker, used for BiSR and SIP-only
    """
    arg_clz = SIPAttackerArguments

    def __init__(self):
        super().__init__()
        self.inverter_b2tr = None
        self.inverter_tr2t = None

    def parse_arguments(self, args, prefix: str):
        res: SIPAttackerArguments = super().parse_arguments(args, prefix)
        if res.dataset is None or len(res.dataset) == 0:
            res.dataset = args.dataset
        if res.target_model_name is None or len(res.target_model_name) == 0:
            res.target_model_name = args.model_name
        if res.target_model_load_bits < 0:
            res.target_model_load_bits = args.load_bits
        res.target_dataset = args.dataset
        res.target_system_sps = args.split_points
        if res.b2tr_layer < 0:
            res.b2tr_layer = int(res.target_system_sps.split('-')[0])
        if res.tr2t_layer < 0:
            res.tr2t_layer = int(res.target_system_sps.split('-')[1])
        if res.model == 'vit':
            res.larger_better = False
        return res

    def load_attacker(self, args, aargs: arg_clz, llm: SplitWrapperModel = None,
                      tokenizer: Tokenizer = None):
        self.inverter_b2tr, self.inverter_tr2t = get_sip_inverter(aargs)
        for choice in ['b2tr', 'tr2t']:
            if getattr(aargs, f'{choice}_enable') and getattr(self, f'inverter_{choice}') is None:
                print(f'Failed to load inverter for {choice}, start training')
                func = train_inversion_model
                if aargs.model.startswith('moe'):
                    func = train_inversion_model_moe
                parser = PrefixArgumentParser(prefix='sip_training', dataclass_types=[InversionModelTrainingArgument])
                training_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
                print(f'Training {choice} inverter using {func}')
                with ParamRestored(llm, llm.param_keeper, ['bottom', 'trunk', 'top'], key='pretrained',
                                   write_back=False, disable_inter_collection=False):
                    with FLConfigHolder(llm):
                        training = llm.training
                        func(llm, tokenizer, aargs, b2tr=(choice == 'b2tr'), training_args=training_args)
                        llm.train(training)
                self.inverter_b2tr, self.inverter_tr2t = get_sip_inverter(aargs)
                assert getattr(self, f'inverter_{choice}') is not None

    def attack(self, args, aargs: arg_clz, llm: SplitWrapperModel, tokenizer: Tokenizer,
               simulator: SFLSimulator, batch, b2tr_inter,
               tr2t_inter, all_inters, init=None):
        attacked_result = {}
        encoder_inter = all_inters.get('encoder', None)
        encoder_inter = None if encoder_inter is None else encoder_inter.fx.to(llm.device)
        with torch.no_grad():
            for type, atk in zip(['tr2t', 'b2tr'], [self.inverter_tr2t, self.inverter_b2tr]):
                if atk is None or not getattr(aargs, f'{type}_enable'):
                    continue
                if aargs.attack_all_layers:
                    for idx, inter in all_inters.items():
                        if not isinstance(idx, int):
                            continue
                        if llm.type == 'encoder-decoder':
                            attacked = atk(torch.concat([encoder_inter.fx.to(
                                simulator.device), inter.fx.to(atk.device)], dim=1))
                        else:
                            attacked = atk(inter.fx.to(atk.device))
                        attacked_result[f'{type}_{idx}'] = attacked
                else:
                    if type == 'b2tr':
                        if aargs.b2tr_target_layer >= 0:
                            inter = all_inters[aargs.b2tr_target_layer - 1]
                        else:
                            inter = b2tr_inter
                    else:
                        if aargs.tr2t_target_layer >= 0:
                            inter = all_inters[aargs.tr2t_target_layer - 1]
                        else:
                            inter = tr2t_inter
                    all_i = {k: (x.fx.shape, x.type) for k, x in all_inters.items()}
                    if llm.type == 'encoder-decoder':
                        attacked = atk(torch.concat([encoder_inter.fx.to(
                            simulator.device), inter.fx.to(atk.device)], dim=1))
                    else:
                        attacked = atk(inter.fx.to(atk.device))
                    attacked_result[type] = attacked

        return attacked_result


def get_sip_inverter(dra_config: SIPAttackerArguments):
    if not dra_config.b2tr_enable and not dra_config.tr2t_enable:
        return None, None
    dataset = dra_config.dataset
    if dataset is None:
        dataset = dra_config.target_dataset

    prefix = dra_config.prefix
    model_name = dra_config.target_model_name
    if required_quantization(model_name):
        model_name += f"-{dra_config.target_model_load_bits}bits"
    attacker_path = dra_config.path + f'{model_name}/{dataset}/'
    matches = []
    if not os.path.exists(attacker_path):
        return None, None
    for d in os.listdir(attacker_path):
        pattern = f'{get_dra_train_label(dataset)}*{dra_config.train_frac:.3f}'
        if ',' in dra_config.dataset:
            pattern = f'Tr{dra_config.train_frac:.3f}'
        if d.startswith(pattern):
            matches.append(os.path.join(attacker_path, d) + '/')
    if len(matches) == 0:
        return None, None
    inverter_paths = [None, None]
    for matched_p in matches:
        p = matched_p + f'{dra_config.model}'
        for idx, choice in enumerate(['b2tr', 'tr2t']):
            if getattr(dra_config, f'{choice}_enable', False):
                sp = int(getattr(dra_config, f'{choice}_target_layer'))
                if getattr(dra_config, f'{choice}_layer', -1) >= 0:
                    sp = dra_config.b2tr_layer
                p += f'/layer{sp}/' + prefix
                if not os.path.exists(p):
                    p = None
                else:
                    l = sorted(list(os.listdir(p)), key=lambda x: float(x.split('_')[-1]))[
                        -1 if dra_config.larger_better else 0]
                    p = os.path.join(p, l)
                    if not os.path.exists(p):
                        p = None
            if inverter_paths[idx] is None:
                inverter_paths[idx] = p

    attacker, attacker2 = None, None
    print(f'Loading inverter from {inverter_paths}')
    if inverter_paths[0]:
        attacker = get_inverter_class(dra_config.model).from_pretrained(inverter_paths[0])
    if inverter_paths[1]:
        attacker2 = get_inverter_class(dra_config.model).from_pretrained(inverter_paths[1])
    return attacker, attacker2
