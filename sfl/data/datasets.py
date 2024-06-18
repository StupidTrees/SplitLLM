import ast

import pandas as pd
import torch
from datasets import load_dataset, Dataset

from sfl import config
from sfl.data.base import FedDataset


class PIQAFedDataset(FedDataset):
    """
    PIQA Dataset
    """

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        super().__init__(tokenizer, client_ids, dataset=load_dataset(config.dataset_cache_dir + 'piqa'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac)

    def _format(self, example):
        q = "### Question: " + example["goal"]
        a = "\n### Solution: " + example["sol1"]
        return {'q': q, 'a': a,
                'input': q + a}

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
            return {'input_ids': input['input_ids'], 'attention_mask': input['attention_mask']
                , 'labels': input['input_ids']}
        else:
            return {'input_ids': input['input_ids'],
                    'attention_mask': input['attention_mask'],
                    'input_text': texts,
                    'q_ids': input_q['input_ids'],
                    'q_att_mask': input_q['attention_mask'],
                    'a_ids': input_a['input_ids'],
                    'a_att_mask': input_a['attention_mask'],
                    'q_text': qs,
                    'a_text': as_,
                    'labels': labels}


class PIQAMiniFedDataset(PIQAFedDataset):
    """
    PIQA数据集
    """

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


class GSM8KFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        super().__init__(tokenizer, client_ids,
                         load_dataset(config.dataset_cache_dir + 'gsm8k', 'main'),
                         types=['train', 'test'], shrink_frac=shrink_frac)

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


class DialogSumFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        super().__init__(tokenizer, client_ids,
                         load_dataset(config.dataset_cache_dir + 'dialogsum'),
                         ['train', 'test', 'validation'],
                         shrink_frac)

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


class CodeAlpacaFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        super().__init__(tokenizer, client_ids, load_dataset(config.dataset_cache_dir + 'CodeAlpaca_20K'),
                         ['train', 'test'],
                         shrink_frac)

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


class IMDBFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        super().__init__(tokenizer, client_ids,
                         load_dataset(config.dataset_cache_dir + 'imdb'),
                         ['train', 'test', 'unsupervised'],
                         shrink_frac, num_labels=2)

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


class WikiTextFedDataset(FedDataset):

    def _format(self, example):
        pass

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac=0.3):
        dataset = load_dataset(config.dataset_cache_dir + 'wikitext', 'wikitext-2-v1')
        types = ['train', 'test', 'validation']
        super().__init__(tokenizer, client_ids, dataset, types, shrink_frac)

    def _pre_process(self, ds, batch_size):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"])

        tokenized_datasets = ds.map(
            tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
        )
        block_size = 128

        def group_texts(examples):
            # 连接所有文本。
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # 我们丢弃小的余数，但如果模型支持的话，您可以添加填充
            # 在这一点上，就像在所有事情上一样，我们建议您跟随自己的内心
            total_length = (total_length // block_size) * block_size
            # 按 max_len 分割。
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

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.df = pd.read_csv(config.dataset_cache_dir + 'sensi/sensi.csv')
        dataset = {
            'train': Dataset.from_pandas(self.df[self.df['type'] == 'train']),
            'validation': Dataset.from_pandas(self.df[self.df['type'] == 'validation']),
            'test': Dataset.from_pandas(self.df[self.df['type'] == 'test'])
        }
        super().__init__(tokenizer, client_ids, dataset, ['train', 'validation', 'test'], shrink_frac)


class SensiReplacedFedDataset(FedDataset):

    def _format(self, example):
        return {'input': example['sani_gpt4']}

    def _col_fun(self, batch, max_seq_len=-1, **kwargs):
        texts = [b['input'] for b in batch]
        max_len = 300 if max_seq_len < 0 else max_seq_len
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {'input_ids': input['input_ids'],
                'q_ids': input['input_ids'],
                'a_ids': input['input_ids'],
                'q_att_mask': input['attention_mask'],
                'attention_mask': input['attention_mask'],
                'input_text': texts}

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.df = pd.read_csv(config.dataset_cache_dir + 'sensi/sensi.csv')
        dataset = {
            'train': Dataset.from_pandas(self.df[self.df['type'] == 'train']),
            'validation': Dataset.from_pandas(self.df[self.df['type'] == 'validation']),
            'test': Dataset.from_pandas(self.df[self.df['type'] == 'test'])
        }
        super().__init__(tokenizer, client_ids, dataset, ['train', 'validation', 'test'], shrink_frac)


class SensiMaskedFedDataset(SensiReplacedFedDataset):

    def _format(self, example):
        return {'input': example['sani_label']}


class HC3CNFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids: list[str]):
        dataset = load_dataset('HC3-Chinese')
        super().__init__(tokenizer, client_ids, dataset, ['baike', 'open_qa', 'finance'])

    def _format(self, example):
        return {'input': example['question']}


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
