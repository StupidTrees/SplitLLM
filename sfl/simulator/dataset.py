from abc import ABC, abstractmethod

from datasets import load_dataset, disable_progress_bar
from torch.utils.data import DataLoader

from sfl.utils import random_slicing


class FedDataset(ABC):
    """
    联邦数据集
    """

    def __init__(self, client_ids: list[str]):
        self.client_ids = client_ids

    @abstractmethod
    def get_dataloader(self, client_id, batch_size=2, type='train'):
        raise NotImplementedError


class PIQAFedDataset(FedDataset):
    """
    PIQA数据集
    """

    def __init__(self, tokenizer, client_ids: list[str], shrink_frac: float = 0.3):
        super().__init__(client_ids)
        self.dataset = load_dataset('piqa')
        self.tokenizer = tokenizer
        # self.dataset['train'] = self.dataset['train'].select(range(330))
        self.client_data_indices = {}
        for type in ['train', 'test', 'validation']:
            self.dataset[type] = self.dataset[type].select(range(int(len(self.dataset[type]) * shrink_frac)))
            sliced = random_slicing(range(len(self.dataset[type])), len(client_ids), sgm=0.15)
            disable_progress_bar()
            self.client_data_indices[type] = {cid: sliced[i] for i, cid in enumerate(client_ids)}

    def get_dataloader(self, client_id, batch_size=2, type='validation'):
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
