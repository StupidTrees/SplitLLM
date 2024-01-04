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
        self.dataset = dataset
        for type in types:
            self.dataset[type] = self.dataset[type].select(range(int(len(self.dataset[type]) * shrink_frac)))
            sliced = random_slicing(range(len(self.dataset[type])), len(client_ids), sgm=0.15)
            disable_progress_bar()
            self.client_data_indices[type] = {cid: sliced[i] for i, cid in enumerate(client_ids)}

    @abstractmethod
    def get_dataloader(self, client_id, batch_size=2, type='train'):
        raise NotImplementedError


class PIQAFedDataset(FedDataset):
    """
    PIQA数据集
    """

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.tokenizer = tokenizer
        super().__init__(client_ids, dataset=load_dataset('piqa'), types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac)

    def get_dataloader(self, client_id, batch_size=1, type='validation'):
        ds = self.dataset[type].select(self.client_data_indices[type][client_id])

        def encode(examples):
            text = examples["goal"] + " Solution:" + examples["sol1"]
            input = self.tokenizer(text, padding="max_length")
            return {'input_ids': input['input_ids'],
                    'input_att_mask': input['attention_mask'],
                    'input_text': text}

        ds = ds.map(encode)
        ds.set_format(type="torch",
                      columns=["input_ids", "input_att_mask", "input_text"])
        loader = DataLoader(ds, batch_size=batch_size)
        return loader


class GSM8KFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.tokenizer = tokenizer
        super().__init__(client_ids, load_dataset('gsm8k', 'main', cache_dir=config.dataset_cache_dir),
                         types=['train', 'test'], shrink_frac=shrink_frac)

    def get_dataloader(self, client_id, batch_size=1, type='train'):
        ds = self.dataset[type].select(self.client_data_indices[type][client_id])

        def encode(examples):
            text = "Question:" + examples["question"] + " Answer:" + examples["answer"]
            input = self.tokenizer(text, padding="max_length")
            return {'input_ids': input['input_ids'],
                    'input_att_mask': input['attention_mask'],
                    'input_text': text}

        ds = ds.map(encode)
        ds.set_format(type="torch",
                      columns=["input_ids", "input_att_mask", "input_text"])
        loader = DataLoader(ds, batch_size=batch_size)
        return loader


class DialogSumFedDataset(FedDataset):

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        self.tokenizer = tokenizer
        super().__init__(client_ids, load_dataset('knkarthick/dialogsum', cache_dir=config.dataset_cache_dir), ['train','test','validation'],
                         shrink_frac)

    def get_dataloader(self, client_id, batch_size=1  , type='train'):
        ds = self.dataset[type].select(self.client_data_indices[type][client_id])

        def encode(examples):
            text = "Dialogue:" + examples["dialogue"] + " Summary:" + examples["summary"]
            input = self.tokenizer(text, padding="max_length")
            return {'input_ids': input['input_ids'],
                    'input_att_mask': input['attention_mask'],
                    'input_text': text}

        ds = ds.map(encode)
        ds.set_format(type="torch",
                      columns=["input_ids", "input_att_mask", "input_text"])
        loader = DataLoader(ds, batch_size=batch_size)
        return loader
