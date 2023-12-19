import random
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

    def __init__(self, tokenizer, client_ids: list[str]):
        super().__init__(client_ids)
        self.dataset = load_dataset('piqa')
        self.tokenizer = tokenizer
        self.dataset['train'] = self.dataset['train'].select(range(300))
        sliced = random_slicing(range(300), len(client_ids), sgm=0.15)
        disable_progress_bar()
        self.client_data_indices = {cid: sliced[i] for i, cid in enumerate(client_ids)}

    def get_dataloader(self, client_id, batch_size=2, type='train'):
        ds = self.dataset[type].select(self.client_data_indices[client_id])

        def encode(examples):
            input = self.tokenizer(examples["goal"], truncation=True, padding="max_length")
            output = self.tokenizer(examples["sol1"], truncation=True, padding="max_length")
            return {'input_ids': input['input_ids'], 'input_att_mask': input['attention_mask'],
                    'output_ids': output['input_ids'], 'output_att_mask': output['attention_mask']}

        ds = ds.map(encode)
        ds.set_format(type="torch", columns=["input_ids", "input_att_mask", "output_ids", "output_att_mask"])
        loader = DataLoader(ds, batch_size=batch_size)
        return loader
