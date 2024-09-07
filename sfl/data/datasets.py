import ast
from copy import deepcopy

import pandas as pd
import torch
from datasets import load_dataset, Dataset
from trl import DataCollatorForCompletionOnlyLM

from sfl import config
from sfl.data.base import FedDataset
from sfl.utils.exp import register_dataset


@register_dataset('piqa')
class PIQAFedDataset(FedDataset):
    """
    PIQA Dataset
    """

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids, dataset=load_dataset(config.dataset_cache_dir + 'piqa'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)
        self.q_temp = "### Question:\n"
        self.a_temp = "### Solution:\n"
        if self.completion_only:
            response_template_ids = tokenizer.encode('\n' + self.a_temp, add_special_tokens=False)[2:]
            self.co_collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids,
                                                               tokenizer=tokenizer)

    def _format(self, example):
        q = self.q_temp + example["goal"]
        a = self.a_temp + example["sol1"]
        return {'q': q, 'a': a,
                'input': q + '\n' + a}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        qs = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')  # for batch_size testing
        input_q = self.tokenizer(qs, padding=True, truncation=True, return_tensors='pt')
        input_a = self.tokenizer(as_, padding=True, truncation=True, return_tensors='pt')
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        if not extra_info:
            res_dict = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask']
                , 'labels': input['input_ids']}
        else:
            res_dict = {'input_ids': input['input_ids'],
                        'attention_mask': input['attention_mask'],
                        'input_text': texts,
                        'q_ids': input_q['input_ids'],
                        'q_att_mask': input_q['attention_mask'],
                        'a_ids': input_a['input_ids'],
                        'a_att_mask': input_a['attention_mask'],
                        'q_text': qs,
                        'a_text': as_,
                        'labels': labels}
        if self.completion_only:
            res_list_tmp = []
            for i in range(len(batch)):
                res_list_tmp.append({'input_ids': res_dict['input_ids'][i],
                                     'attention_mask': res_dict['attention_mask'][i],
                                     'labels': deepcopy(res_dict['input_ids'][i])})
            res_list = self.co_collator(res_list_tmp)
            res_dict_tmp = res_list
            res_dict['input_ids'] = res_dict_tmp['input_ids']
            res_dict['attention_mask'] = res_dict_tmp['attention_mask']
            res_dict['labels'] = res_dict_tmp['labels']
        return res_dict


@register_dataset('piqa-mini')
class PIQAMiniFedDataset(PIQAFedDataset):
    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        qs = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        max_len = 128 if max_seq_len < 0 else max_seq_len
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                               max_length=max_len)  # for batch_size testing
        input_q = self.tokenizer(qs, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        input_a = self.tokenizer(as_, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        return {'input_ids': input['input_ids'],
                'attention_mask': input['attention_mask'],
                'input_text': texts,
                'q_ids': input_q['input_ids'],
                'q_att_mask': input_q['attention_mask'],
                'a_ids': input_a['input_ids'],
                'a_att_mask': input_a['attention_mask'],
                'q_text': qs,
                'a_text': as_, 'labels': labels}


@register_dataset('stsb')
class STSBFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids, dataset=load_dataset(config.dataset_cache_dir + 'stsb'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)
        self.q_temp = "### Sentence1:\n"
        self.a_temp = "### Sentence2:\n"

    def _format(self, example):
        q = self.q_temp + example["sentence1"]
        a = self.a_temp + example["sentence2"]
        return {'input': q + '\n' + a}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        kwargs = {'padding': True, 'truncation': True, 'return_tensors': 'pt'}
        if self.uni_length > 0:
            kwargs['max_length'] = self.uni_length
            kwargs['padding'] = 'max_length'
        input = self.tokenizer(texts, **kwargs)  # for batch_size testing
        labels = [b['score'] for b in batch]
        labels = torch.tensor(labels)
        if not extra_info:
            res_dict = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask']
                , 'labels': input['input_ids']}
        else:
            res_dict = {'input_ids': input['input_ids'],
                        'attention_mask': input['attention_mask'],
                        'input_text': texts,
                        'labels': labels}
        return res_dict


@register_dataset('qnli')
class QNLIFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids, dataset=load_dataset(config.dataset_cache_dir + 'qnli'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)
        self.q_temp = "### Question:\n"
        self.a_temp = "### Solution:\n"

    def _format(self, example):
        q = self.q_temp + example["text1"]
        a = self.a_temp + example["text2"]
        return {'q': q, 'a': a,
                'input': q + '\n' + a}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        qs = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        # pad&truncate all sentence to the same length (self.uni_length)
        kwargs = {'padding': True, 'truncation': True, 'return_tensors': 'pt'}
        if self.uni_length > 0:
            kwargs['max_length'] = self.uni_length
            kwargs['padding'] = 'max_length'
        input = self.tokenizer(texts, **kwargs)  # for batch_size testing
        input_q = self.tokenizer(qs, **kwargs)
        input_a = self.tokenizer(as_, **kwargs)
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        if not extra_info:
            res_dict = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask']
                , 'labels': input['input_ids']}
        else:
            res_dict = {'input_ids': input['input_ids'],
                        'attention_mask': input['attention_mask'],
                        'input_text': texts,
                        'q_ids': input_q['input_ids'],
                        'q_att_mask': input_q['attention_mask'],
                        'a_ids': input_a['input_ids'],
                        'a_att_mask': input_a['attention_mask'],
                        'q_text': qs,
                        'a_text': as_,
                        'labels': labels}
        return res_dict


@register_dataset('mrpc')
class MRPCFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids, dataset=load_dataset(config.dataset_cache_dir + 'mrpc'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)
        self.q_temp = "### Question:\n"
        self.a_temp = "### Solution:\n"

    def _format(self, example):
        q = self.q_temp + example["text1"]
        a = self.a_temp + example["text2"]
        return {'q': q, 'a': a,
                'input': q + '\n' + a}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        qs = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        kwargs = {'padding': True, 'truncation': True, 'return_tensors': 'pt'}
        if self.uni_length > 0:
            kwargs['max_length'] = self.uni_length
            kwargs['padding'] = 'max_length'
        input = self.tokenizer(texts, **kwargs)  # for batch_size testing
        input_q = self.tokenizer(qs, **kwargs)
        input_a = self.tokenizer(as_, **kwargs)
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        if not extra_info:
            res_dict = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask']
                , 'labels': input['input_ids']}
        else:
            res_dict = {'input_ids': input['input_ids'],
                        'attention_mask': input['attention_mask'],
                        'input_text': texts,
                        'q_ids': input_q['input_ids'],
                        'q_att_mask': input_q['attention_mask'],
                        'a_ids': input_a['input_ids'],
                        'a_att_mask': input_a['attention_mask'],
                        'q_text': qs,
                        'a_text': as_,
                        'labels': labels}
        return res_dict


@register_dataset('cola', dra_train_label='test')
class CoLAFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids, dataset=load_dataset(config.dataset_cache_dir + 'cola'),
                         types=['train', 'test'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        return {'input': example['text']}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        kwargs = {'padding': True, 'truncation': True, 'return_tensors': 'pt'}
        if self.uni_length > 0:
            kwargs['max_length'] = self.uni_length
            kwargs['padding'] = 'max_length'
        input = self.tokenizer(texts, **kwargs)  # for batch_size testing
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        res_dict = {'input_ids': input['input_ids'],
                    'attention_mask': input['attention_mask'],
                    'input_text': texts,
                    'labels': labels}
        return res_dict


@register_dataset('rte')
class RTEFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids, dataset=load_dataset(config.dataset_cache_dir + 'rte'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)
        self.q_temp = "### Question:\n"
        self.a_temp = "### Solution:\n"

    def _format(self, example):
        q = self.q_temp + example["text1"]
        a = self.a_temp + example["text2"]
        return {'q': q, 'a': a,
                'input': q + '\n' + a}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        qs = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        kwargs = {'padding': True, 'truncation': True, 'return_tensors': 'pt'}
        if self.uni_length > 0:
            kwargs['max_length'] = self.uni_length
            kwargs['padding'] = 'max_length'
        input = self.tokenizer(texts, **kwargs)  # for batch_size testing
        input_q = self.tokenizer(qs, **kwargs)
        input_a = self.tokenizer(as_, **kwargs)
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        if not extra_info:
            res_dict = {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask']
                , 'labels': input['input_ids']}
        else:
            res_dict = {'input_ids': input['input_ids'],
                        'attention_mask': input['attention_mask'],
                        'input_text': texts,
                        'q_ids': input_q['input_ids'],
                        'q_att_mask': input_q['attention_mask'],
                        'a_ids': input_a['input_ids'],
                        'a_att_mask': input_a['attention_mask'],
                        'q_text': qs,
                        'a_text': as_,
                        'labels': labels}
        return res_dict


@register_dataset('gsm8k', dra_train_label='test')
class GSM8KFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids,
                         load_dataset(config.dataset_cache_dir + 'gsm8k', 'main'),
                         types=['train', 'test'], shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        q = "### Question: " + example['question']
        a = "### Answer: " + example['answer']
        return {'q': q, 'a': a,
                'input': q + "\n" + a}

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        max_len = 300 if max_seq_len < 0 else max_seq_len
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                               max_length=max_len)  # 300, glm 256
        qs_ = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        input_q = self.tokenizer(qs_, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        input_a = self.tokenizer(as_, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {'input_ids': input['input_ids'],
                'attention_mask': input['attention_mask'],
                'input_text': texts,
                'q_ids': input_q['input_ids'],
                'q_att_mask': input_q['attention_mask'],
                'a_ids': input_a['input_ids'],
                'a_att_mask': input_a['attention_mask'],
                'q_text': qs_,
                'a_text': as_}


@register_dataset('dialogsum')
class DialogSumFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids,
                         load_dataset(config.dataset_cache_dir + 'dialogsum'),
                         ['train', 'test', 'validation'],
                         shrink_frac, **kwargs)

    def _format(self, example):
        q = "### Dialogue: " + example["dialogue"]
        a = "### Summary: " + example["summary"]
        return {'q': q, 'a': a,
                'input': q + "\n" + a}

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        max_len = 400 if max_seq_len < 0 else max_seq_len
        input = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len,  # chatglm 256
                               return_tensors='pt')
        qs_ = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        input_q = self.tokenizer(qs_, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        input_a = self.tokenizer(as_, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {'input_ids': input['input_ids'],
                'attention_mask': input['attention_mask'],
                'input_text': texts,
                'q_ids': input_q['input_ids'],
                'q_att_mask': input_q['attention_mask'],
                'a_ids': input_a['input_ids'],
                'a_att_mask': input_a['attention_mask'],
                'q_text': qs_,
                'a_text': as_}


@register_dataset('codealpaca', dra_train_label='test')
class CodeAlpacaFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids, load_dataset(config.dataset_cache_dir + 'CodeAlpaca_20K'),
                         ['train', 'test'],
                         shrink_frac, **kwargs)

    def _format(self, example):
        q = "### Question: " + example['prompt']
        a = "### Answer: " + example['completion']
        return {'q': q, 'a': a,
                'input': q + "\n" + a}

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        max_len = 400 if max_seq_len < 0 else max_seq_len
        input = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len,  # 400, chatglm256
                               return_tensors='pt')
        qs_ = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        input_q = self.tokenizer(qs_, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        input_a = self.tokenizer(as_, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {'input_ids': input['input_ids'],
                'attention_mask': input['attention_mask'],
                'input_text': texts,
                'q_ids': input_q['input_ids'],
                'q_att_mask': input_q['attention_mask'],
                'a_ids': input_a['input_ids'],
                'a_att_mask': input_a['attention_mask'],
                'q_text': qs_,
                'a_text': as_}


@register_dataset('imdb', dra_train_label='unsupervised')
class IMDBFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        super().__init__(tokenizer, client_ids,
                         load_dataset(config.dataset_cache_dir + 'imdb'),
                         ['train', 'test', 'unsupervised'],
                         shrink_frac, num_labels=2, **kwargs)

    def _format(self, example):
        return {'input': example['text']}

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        max_len = 512 if max_seq_len < 0 else max_seq_len
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        # convert labels to tensor
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        return {'input_ids': input['input_ids'],
                'attention_mask': input['attention_mask'],
                'input_text': texts, 'labels': labels}


@register_dataset('wikitext')
class WikiTextFedDataset(FedDataset):

    def _format(self, example):
        pass

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac=0.3, **kwargs):
        dataset = load_dataset(config.dataset_cache_dir + 'wikitext', 'wikitext-2-v1')
        types = ['train', 'test', 'validation']
        super().__init__(tokenizer, client_ids, dataset, types, shrink_frac, **kwargs)

    def _pre_process(self, ds, batch_size):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"])

        tokenized_datasets = ds.map(
            tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
        )
        block_size = 128
        if self.uni_length > 0:
            block_size = self.uni_length

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            result["input_text"] = [self.tokenizer.decode(ii) for ii in result["input_ids"]]
            result["attention_mask"] = result["attention_mask"]
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=batch_size,
            num_proc=4,
        )
        lm_datasets.set_format(type="torch")
        return lm_datasets

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        res = {}
        for k in batch[0].keys():
            ls = [x[k] for x in batch]
            if isinstance(ls[0], torch.Tensor):
                res[k] = torch.stack(ls)
            else:
                res[k] = ls
        return res


@register_dataset('wikitext-103')
class WikiText103FedDataset(WikiTextFedDataset):
    def __init__(self, tokenizer, client_ids: list[str], shrink_frac=0.3, **kwargs):
        dataset = load_dataset(config.dataset_cache_dir + 'wikitext', 'wikitext-103-v1')
        types = ['train', 'test', 'validation']
        FedDataset.__init__(self, tokenizer, client_ids, dataset, types, shrink_frac, **kwargs)


@register_dataset('sensimarked')
class SensiMarkedFedDataset(FedDataset):

    def _format(self, example):
        return {'input': example['content'], 'entities': ast.literal_eval(example['entity'])}

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        max_len = 300 if max_seq_len < 0 else max_seq_len
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                               max_length=max_len)  # 300, ChatgGLM bs=2时降低
        mask = torch.zeros_like(input['input_ids'])
        for sp, sample in enumerate(batch):
            seq = input['input_ids'][sp].numpy().tolist()
            r = self.tokenizer(sample['entities'], add_special_tokens=False)
            for subseq in r.input_ids:
                for i in range(len(seq) - len(subseq) + 1):
                    if seq[i:i + len(subseq)] == subseq:
                        mask[sp, i:i + len(subseq)] = 1

        return {'input_ids': input['input_ids'],
                'q_ids': input['input_ids'],
                'a_ids': input['input_ids'],
                'q_att_mask': input['attention_mask'],
                'attention_mask': input['attention_mask'],
                'input_text': texts, 'entities': [b['entity'] for b in batch],
                'input_santi_mask': mask}

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        self.df = pd.read_csv(config.dataset_cache_dir + 'sensi/sensi.csv')
        dataset = {
            'train': Dataset.from_pandas(self.df[self.df['type'] == 'train']),
            'validation': Dataset.from_pandas(self.df[self.df['type'] == 'validation']),
            'test': Dataset.from_pandas(self.df[self.df['type'] == 'test'])
        }
        super().__init__(tokenizer, client_ids, dataset, ['train', 'validation', 'test'], shrink_frac, **kwargs)


@register_dataset('sensireplaced')
class SensiReplacedFedDataset(FedDataset):

    def _format(self, example):
        return {'input': example['sani_gpt4']}

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        max_len = 300 if max_seq_len < 0 else max_seq_len
        kwargs = dict(padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        if self.uni_length > 0:
            kwargs['max_length'] = self.uni_length
            kwargs['padding'] = 'max_length'
        input = self.tokenizer(texts, **kwargs)  # for batch_size testing
        return {'input_ids': input['input_ids'],
                'q_ids': input['input_ids'],
                'a_ids': input['input_ids'],
                'q_att_mask': input['attention_mask'],
                'attention_mask': input['attention_mask'],
                'input_text': texts}

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3, **kwargs):
        self.df = pd.read_csv(config.dataset_cache_dir + 'sensi/sensi.csv')
        dataset = {
            'train': Dataset.from_pandas(self.df[self.df['type'] == 'train']),
            'validation': Dataset.from_pandas(self.df[self.df['type'] == 'validation']),
            'test': Dataset.from_pandas(self.df[self.df['type'] == 'test'])
        }
        super().__init__(tokenizer, client_ids, dataset, ['train', 'validation', 'test'], shrink_frac, **kwargs)


@register_dataset('sensimasked')
class SensiMaskedFedDataset(SensiReplacedFedDataset):

    def _format(self, example):
        return {'input': example['sani_label']}


@register_dataset('hc3cn', dra_train_label='baike',dra_test_label='finance')
class HC3CNFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids: list[str]):
        dataset = load_dataset('HC3-Chinese')
        super().__init__(tokenizer, client_ids, dataset, ['baike', 'open_qa', 'finance'])

    def _format(self, example):
        return {'input': example['question']}


@register_dataset('imagewoof',dra_test_label='validation')
class ImageWoofFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 1.0):
        ds = load_dataset(config.dataset_cache_dir + 'imagewoof', '160px')
        super().__init__(tokenizer, client_ids, ds, ['train', 'validation'], shrink_frac)

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        images = [s['image'].convert("RGB") for s in batch]
        labels = torch.tensor([s['label'] for s in batch])
        return {'input': self.tokenizer(images, return_tensors='pt', padding=True)['pixel_values'],
                'labels': labels,
                'image': images}

    def _format(self, example):
        return example

    def _pre_process(self, ds, batch_size):
        return ds
