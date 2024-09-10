import os

from sfl.model.reducer.args import ReducerArgument, ReductionTrainingArguments
from sfl.model.reducer.reducer_models import DimReduction
from sfl.model.reducer.reducer_training import train_reducer
from sfl.utils.args import PrefixArgumentParser
from sfl.utils.exp import required_quantization
from sfl.utils.model import ParamRestored, FLConfigHolder


def _get_reducer_args(sys_args) -> ReducerArgument:
    parser = PrefixArgumentParser(prefix='reducer', dataclass_types=[ReducerArgument])
    red_arg: ReducerArgument = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
    if red_arg.target_model is None or red_arg.target_model == '':
        red_arg.target_model = sys_args.model_name
    if red_arg.target_model_load_bits < 0:
        red_arg.target_model_load_bits = sys_args.load_bits
    if red_arg.dataset is None or red_arg.dataset == '':
        red_arg.dataset = sys_args.dataset
    return red_arg


def _load_dim_reduction(red_arg: ReducerArgument):
    dataset = red_arg.dataset
    model_name = red_arg.target_model
    if required_quantization(model_name):
        model_name += f"-{red_arg.target_model_load_bits}bits"
    mapper_path = red_arg.path + f'{model_name}/{dataset}/'
    matches = []
    if not os.path.exists(mapper_path):
        return None
    for d in os.listdir(mapper_path):
        pattern = f'{red_arg.train_label}*{red_arg.train_frac:.3f}'
        if ',' in dataset:
            pattern = f'Tr{red_arg.train_frac:.3f}'
        if d.startswith(pattern):
            mapper_path = os.path.join(mapper_path, d) + '/'
            matches.append(mapper_path)
    if len(matches) == 0:
        return None
    mapper_path_1 = None
    for attacker_path in matches:
        mapper_path_1 = attacker_path + f'layer{red_arg.layer}/{red_arg.alpha}'
        if not os.path.exists(mapper_path_1):
            mapper_path_1 = None
        else:
            l = sorted(list(os.listdir(mapper_path_1)), key=lambda x: float(x.split('_')[-1]))[0]
            mapper_path_1 = os.path.join(mapper_path_1, l)
            if not os.path.exists(mapper_path_1):
                mapper_path_1 = None
    if mapper_path_1:
        return DimReduction.from_pretrained(mapper_path_1)
    return None


def get_dim_reducer(sys_args, llm, tokenizer) -> DimReduction:
    red_arg = _get_reducer_args(sys_args)
    reducer = _load_dim_reduction(red_arg)
    if reducer is None:
        print(f'No reducer found for {red_arg}, starting training...')
        with (ParamRestored(llm, llm.param_keeper, ['bottom', 'trunk', 'top'], key='pretrained',
                            write_back=False, disable_inter_collection=False)):
            with FLConfigHolder(llm):
                training = llm.training
                parser = PrefixArgumentParser(prefix='reducer_training', dataclass_types=[ReductionTrainingArguments])
                training_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
                train_reducer(llm, tokenizer, red_arg, training_args)
                llm.train(training)
        reducer = _load_dim_reduction(red_arg)
    assert reducer is not None
    return reducer
