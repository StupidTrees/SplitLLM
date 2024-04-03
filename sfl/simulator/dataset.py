import ast
from abc import ABC, abstractmethod

import pandas as pd
import torch
from datasets import load_dataset, disable_progress_bar, Dataset
from torch.utils.data import DataLoader

from sfl import config
from sfl.config import DRA_train_label, DRA_test_label
from sfl.utils.data import random_slicing


class FedDataset(ABC):
    """
    联邦数据集
    """

    def __init__(self, tokenizer, client_ids: list[str], dataset, types: list[str], shrink_frac=1.0,
                 num_labels=0):
        self.tokenizer = tokenizer
        self.client_ids = client_ids
        self.client_data_indices = {}
        self.all_dataset = dataset
        self.dataset = {}
        self.num_labels = num_labels
        for type in types:
            self.dataset[type] = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
            sliced = random_slicing(range(len(self.dataset[type])), len(client_ids), sgm=0.15)
            disable_progress_bar()
            self.client_data_indices[type] = {cid: sliced[i] for i, cid in enumerate(client_ids)}

    def get_dataloader(self, client_id, batch_size=1, type='train'):
        ds = self.dataset[type].select(self.client_data_indices[type][client_id])
        return DataLoader(self._pre_process(ds, batch_size),
                          collate_fn=lambda x: self._col_fun(x),
                          batch_size=batch_size,
                          shuffle=True)

    def get_dataloader_unsliced(self, batch_size=2, type='train', shrink_frac=1.0, further_test_split=None):
        ds = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
        if further_test_split is not None:
            ds_split = ds.train_test_split(shuffle=True, test_size=further_test_split)
            return DataLoader(self._pre_process(ds_split['train'], batch_size),
                              collate_fn=lambda x: self._col_fun(x),
                              batch_size=batch_size,
                              shuffle=True), \
                   DataLoader(self._pre_process(ds_split['test'], batch_size),
                              collate_fn=lambda x: self._col_fun(x),
                              batch_size=batch_size, shuffle=True)
        return DataLoader(self._pre_process(ds, batch_size), batch_size=batch_size, shuffle=True,
                          collate_fn=lambda x: self._col_fun(x))

    def _pre_process(self, ds, batch_size):
        ds = ds.map(lambda x: self._format(x), batched=False)
        ds.set_format(type="torch")
        return ds

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts}

    @abstractmethod
    def _format(self, example):
        raise NotImplementedError


class CombinedDataLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders)

    def __iter__(self):
        # 创建迭代器列表
        iterators = [iter(dataloader) for dataloader in self.dataloaders]

        while iterators:
            # 随机选择一个DataLoader
            idx = torch.randint(len(iterators), (1,)).item()
            try:
                # 获取数据
                data = next(iterators[idx])
                data['type'] = idx
                yield data
            except StopIteration:
                # 如果这个DataLoader已经没有数据了，就从列表中移除
                del iterators[idx]


class MixtureFedDataset(FedDataset):

    def _format(self, example):
        pass

    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, dataset_names=None, dataset_classes=None):
        super().__init__(tokenizer, client_ids, None, [], shrink_frac)
        if dataset_names is None:
            dataset_names = []
            dataset_classes = []
        self.fed_datasets = []
        self.dataset_names = dataset_names
        for cls in dataset_classes:
            self.fed_datasets.append(cls(tokenizer, client_ids, shrink_frac))

    def get_dataloader(self, client_id, batch_size=1, type='train'):
        return CombinedDataLoader(*[ds.get_dataloader(client_id, batch_size, type) for ds in self.fed_datasets])

    def get_dataloader_unsliced(self, batch_size=2, type=None, shrink_frac=1.0, further_test_split=None):
        train_loaders = []
        test_loaders = []
        for nm, ds in zip(self.dataset_names, self.fed_datasets):
            if DRA_train_label[nm] == DRA_test_label[nm]:
                d1, d2 = ds.get_dataloader_unsliced(batch_size, DRA_train_label[nm], shrink_frac,
                                                    further_test_split=0.3)
            else:
                d1 = ds.get_dataloader_unsliced(batch_size, DRA_train_label[nm], shrink_frac=shrink_frac)
                d2 = ds.get_dataloader_unsliced(batch_size, DRA_test_label[nm], shrink_frac=shrink_frac)
            train_loaders.append(d1)
            test_loaders.append(d2)
        return CombinedDataLoader(*train_loaders), CombinedDataLoader(*test_loaders)


class PIQAFedDataset(FedDataset):
    """
    PIQA数据集
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

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        qs = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')  # for batch_size testing
        input_q = self.tokenizer(qs, padding=True, truncation=True, return_tensors='pt')
        input_a = self.tokenizer(as_, padding=True, truncation=True, return_tensors='pt')
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts,
                'q_ids': input_q['input_ids'],
                'q_att_mask': input_q['attention_mask'],
                'a_ids': input_a['input_ids'],
                'a_att_mask': input_a['attention_mask'],
                'q_text': qs,
                'a_text': as_, 'labels': labels}


class PIQAMiniFedDataset(PIQAFedDataset):
    """
    PIQA数据集
    """

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        qs = [b['q'] for b in batch]
        as_ = [b['a'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                               max_length=128)  # for batch_size testing
        input_q = self.tokenizer(qs, padding=True, truncation=True, return_tensors='pt', max_length=128)
        input_a = self.tokenizer(as_, padding=True, truncation=True, return_tensors='pt', max_length=128)
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
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

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                               max_length=300)  # 300, glm 256
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts}


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

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, max_length=400,  # chatglm 256
                               return_tensors='pt')
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts}


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

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, max_length=400,  # 400, chatglm256
                               return_tensors='pt')
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts}


class IMDBFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        super().__init__(tokenizer, client_ids,
                         load_dataset(config.dataset_cache_dir + 'imdb'),
                         ['train', 'test', 'unsupervised'],
                         shrink_frac, num_labels=2)

    def _format(self, example):
        return {'input': example['text']}

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        # convert labels to tensor
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
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
            result["input_att_mask"] = result["attention_mask"]
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=batch_size,
            num_proc=4,
        )
        lm_datasets.set_format(type="torch")
        return lm_datasets

    def _col_fun(self, batch):
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

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                               max_length=300)  # 300, ChatgGLM bs=2时降低
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
                'input_att_mask': input['attention_mask'],
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

    def _col_fun(self, batch):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=300)
        return {'input_ids': input['input_ids'],
                'q_ids': input['input_ids'],
                'a_ids': input['input_ids'],
                'q_att_mask': input['attention_mask'],
                'input_att_mask': input['attention_mask'],
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

    def _col_fun(self, batch):
        images = [s['image'].convert("RGB") for s in batch]
        labels = torch.tensor([s['label'] for s in batch])
        return {'input': self.tokenizer(images, return_tensors='pt', padding=True)['pixel_values'],
                'labels': labels,
                'image': images}

    def _format(self, example):
        return example

    def _pre_process(self, ds, batch_size):
        return ds
