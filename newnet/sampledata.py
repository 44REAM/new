# pylint: disable=redefined-outer-name

import numpy as np
from torch.utils.data import Dataset

from .utils import get_dataloader


class SampleDataset3D(Dataset):

    def __init__(self, transforms = None, n_sample=100, channels=1, width=64, height=64, deep=12):
        np.random.seed(52)
        self.data = np.random.randn(
            n_sample, channels, width, height, deep).astype(np.float32)
        self.labels = np.array([np.random.randint(0, 2)
                                for x in range(n_sample)]).astype(np.float32)
        self.list_ids = np.array([i for i in range(n_sample)])
        self.transforms = transforms
        self.dim = 3

    def get_data(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        idx = self.list_ids[index]

        X = self.get_data(idx)
        y = self.labels[index]

        return X, y

def sample_dataloader3d(config, validation_split=0.2, transforms=None,
                 shuffle_dataset=True, random_seed=42, n_sample = 10):

    dataset = SampleDataset3D(transforms = transforms, n_sample= n_sample)
    dataloader = get_dataloader(dataset, config.BATCHSIZE,
                                shuffle_dataset=shuffle_dataset, random_seed=random_seed)
    return dataloader


class SampleDataset2D(Dataset):

    def __init__(self, transforms = None, n_sample=100, channels=3, width=224, height=244):
        np.random.seed(52)
        self.data = np.random.randn(
            n_sample, width, height, channels).astype(np.float32)
        self.labels = np.array([np.random.randint(0, 2)
                                for x in range(n_sample)]).astype(np.float32)
        self.list_ids = np.array([i for i in range(n_sample)])
        self.transforms = transforms
        self.dim = 2

    def get_data(self, idx):

        return self.data[idx]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        idx = self.list_ids[index]

        X = self.get_data(idx)
        #X = self.transforms(X)
        y = self.labels[index]

        return X, y

def sample_dataloader2d(config, validation_split=0.2, transforms=None,
                 shuffle_dataset=True, random_seed=42):

    dataset = SampleDataset2D(transforms)
    dataloader = get_dataloader(dataset, config.BATCHSIZE,
                                shuffle_dataset=shuffle_dataset, random_seed=random_seed)
    return dataloader

if __name__ == '__main__':
    from ..config import Config
    from .transformers import Transforms
    from .utils import show, print_dataloader


    dataloader = sample_dataloader3d(Config)


    print('train')
    print_dataloader(dataloader['train'])
    print('val')
    print_dataloader(dataloader['val'])
    print('test')
    print_dataloader(dataloader['test'])

    

