from abc import ABC, abstractmethod
from functools import partial

import torch
from datasets import disable_progress_bar
from torch.utils.data import DataLoader

from sfl.config import DRA_train_label, DRA_test_label
from sfl.utils.data import random_slicing


class FedDataset(ABC):
    """
    Federated (Split) Learning Dataset
    """

    def __init__(self, tokenizer, client_ids: list[str], dataset, types: list[str], shrink_frac=1.0,
                 num_labels=0, completion_only=False, uni_length: int = -1):
        self.tokenizer = tokenizer
        self.client_ids = client_ids
        self.client_data_indices = {}
        self.all_dataset = dataset
        self.dataset = {}
        self.completion_only = completion_only
        self.num_labels = num_labels
        self.uni_length = uni_length
        for type in types:
            self.dataset[type] = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
            sliced = random_slicing(range(len(self.dataset[type])), len(client_ids), sgm=0.15)
            disable_progress_bar()
            self.client_data_indices[type] = {cid: sliced[i] for i, cid in enumerate(client_ids)}

    def get_dataloader(self, client_id, batch_size=1, type='train', max_seq_len=-1):
        ds = self.dataset[type].select(self.client_data_indices[type][client_id])
        return DataLoader(self._pre_process(ds, batch_size),
                          collate_fn=lambda x: self._col_fun(x, max_seq_len=max_seq_len),
                          batch_size=batch_size,
                          shuffle=True)

    def as_dataset_and_collator(self, type='train', shrink_frac=1.0):
        ds = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
        ds = self._pre_process(ds, 1)
        return ds, partial(self._col_fun, extra_info=False)

    def get_dataloader_unsliced(self, batch_size=2, type='train', shrink_frac=1.0, further_test_split=None,
                                max_seq_len=-1, shuffle=True):
        ds = self.all_dataset[type].select(range(int(len(self.all_dataset[type]) * shrink_frac)))
        if further_test_split is not None:
            ds_split = ds.train_test_split(shuffle=shuffle, test_size=further_test_split)
            return DataLoader(self._pre_process(ds_split['train'], batch_size),
                              collate_fn=lambda x: self._col_fun(x, max_seq_len=max_seq_len),
                              batch_size=batch_size,
                              shuffle=shuffle), \
                DataLoader(self._pre_process(ds_split['test'], batch_size),
                           collate_fn=lambda x: self._col_fun(x, max_seq_len=max_seq_len),
                           batch_size=batch_size, shuffle=shuffle)
        return DataLoader(self._pre_process(ds, batch_size), batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda x: self._col_fun(x, max_seq_len=max_seq_len))

    def _pre_process(self, ds, batch_size):
        ds = ds.map(lambda x: self._format(x), batched=False)
        ds.set_format(type="torch")
        return ds

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return {'input_ids': input['input_ids'],
                'attention_mask': input['attention_mask'],
                'input_text': texts}

    @abstractmethod
    def _format(self, example):
        raise NotImplementedError


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

    def get_dataloader(self, client_id, batch_size=1, type='train', max_seq_len=-1):
        return CombinedDataLoader(
            *[ds.get_dataloader(client_id, batch_size, type, max_seq_len=max_seq_len) for ds in self.fed_datasets])

    def get_dataloader_unsliced(self, batch_size=2, type=None, shrink_frac=1.0, further_test_split=None,
                                max_seq_len=-1):
        train_loaders = []
        test_loaders = []
        for nm, ds in zip(self.dataset_names, self.fed_datasets):
            if DRA_train_label[nm] == DRA_test_label[nm]:
                d1, d2 = ds.get_dataloader_unsliced(batch_size, DRA_train_label[nm], shrink_frac,
                                                    further_test_split=0.3, max_seq_len=max_seq_len)
            else:
                d1 = ds.get_dataloader_unsliced(batch_size, DRA_train_label[nm], shrink_frac=shrink_frac,
                                                max_seq_len=max_seq_len)
                d2 = ds.get_dataloader_unsliced(batch_size, DRA_test_label[nm], shrink_frac=shrink_frac,
                                                max_seq_len=max_seq_len)
            train_loaders.append(d1)
            test_loaders.append(d2)
        return CombinedDataLoader(*train_loaders), CombinedDataLoader(*test_loaders)


class CombinedDataLoader:
    """
    Combine multiple DataLoaders into one
    """

    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders)

    def __iter__(self):
        # list of iterators
        iterators = [iter(dataloader) for dataloader in self.dataloaders]

        while iterators:
            # randomly select a DataLoader
            idx = torch.randint(len(iterators), (1,)).item()
            try:
                # get the next data from the DataLoader
                data = next(iterators[idx])
                data['type'] = idx
                yield data
            except StopIteration:
                # delete the DataLoader if there is no more data
                del iterators[idx]
