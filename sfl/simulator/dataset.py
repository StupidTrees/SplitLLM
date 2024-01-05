from abc import ABC, abstractmethod

from datasets import load_dataset, disable_progress_bar
from torch.utils.data import DataLoader

from sfl import config
from sfl.utils import random_slicing


class FedDataset(ABC):
    """
    联邦数据集
    """

    def __init__(self, client_ids: list[str], dataset, types: list[str], shrink_frac=1.0):
        self.client_ids = client_ids
        self.client_data_indices = {}
        self.all_dataset = dataset
        self.dataset = {}
        for type in types:
            self.dataset[type] = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
            sliced = random_slicing(range(len(self.dataset[type])), len(client_ids), sgm=0.15)
            disable_progress_bar()
            self.client_data_indices[type] = {cid: sliced[i] for i, cid in enumerate(client_ids)}

    def get_dataloader(self, client_id, batch_size=2, type='train'):
        ds = self.dataset[type].select(self.client_data_indices[type][client_id])

        def encode(examples):
            return self._encode(examples)

        ds = ds.map(encode, batched=True, batch_size=batch_size)
        ds.set_format(type="torch",
                      columns=["input_ids", "input_att_mask", "input_text"])
        loader = DataLoader(ds, batch_size=batch_size)
        return loader

    def get_dataloader_unsliced(self, batch_size=2, type='train', shrink_frac=1.0):
        ds = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))

        def encode(examples):
            return self._encode(examples)

        ds = ds.map(encode, batched=True, batch_size=batch_size)
        ds.set_format(type="torch",
                      columns=["input_ids", "input_att_mask", "input_text"])
        loader = DataLoader(ds, batch_size=batch_size)
        return loader

    @abstractmethod
    def _encode(self, examples):
        raise NotImplementedError


class PIQAFedDataset(FedDataset):
    """
    PIQA数据集
    """

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.tokenizer = tokenizer
        super().__init__(client_ids, dataset=load_dataset('piqa', cache_dir=config.dataset_cache_dir),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac)

    def _encode(self, examples):
        texts = [q + ", Solution: " + a for q, a in zip(examples["goal"], examples["sol1"])]
        input = self.tokenizer(texts, padding=True, truncation=True)
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts}


class GSM8KFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.tokenizer = tokenizer
        super().__init__(client_ids, load_dataset('gsm8k', 'main', cache_dir=config.dataset_cache_dir),
                         types=['train', 'test'], shrink_frac=shrink_frac)

    def _encode(self, examples):
        texts = ["Question: " + q + ", Answer: " + a for q, a in zip(examples["question"], examples["answer"])]
        # text = "Question:" + examples["question"] + " Answer:" + examples["answer"]
        input = self.tokenizer(texts, padding=True, truncation=True)
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts}


class DialogSumFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.tokenizer = tokenizer
        super().__init__(client_ids, load_dataset('knkarthick/dialogsum', cache_dir=config.dataset_cache_dir),
                         ['train', 'test', 'validation'],
                         shrink_frac)

    def _encode(self, examples):
        texts = ["Dialogue: " + q + ", Summary: " + a for q, a in zip(examples["dialogue"], examples["summary"])]
        input = self.tokenizer(texts, padding=True, truncation=True)
        return {'input_ids': input['input_ids'],
                'input_att_mask': input['attention_mask'],
                'input_text': texts}
