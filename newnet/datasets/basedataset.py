from torch.utils.data import Dataset
import numpy as np


class BaseDataset2D(Dataset):
    """Base for dataset.
    This class use for separate training set and validation set, to make
    augmentation only done in training set and not done in validation set.

    Arguments:
        Dataset {[type]} -- dataset class from pytorch
    """

    def __init__(self, dataset, transform=True):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]
        if self.transform:
            if self.dataset.transforms is not None:
                X = self.dataset.transforms(X)
            else:
                pass

        X = np.transpose(X, (2, 0, 1)).astype(np.float32)
        return X, y

class BaseDataset3D(Dataset):
    """Base for dataset.
    This class use for separate training set and validation set, to make
    augmentation only done in training set and not done in validation set.

    Arguments:
        Dataset {[type]} -- dataset class from pytorch
    """

    def __init__(self, dataset, transform=True):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]
        if self.transform:
            if self.dataset.transforms is not None:
                X = self.dataset.transforms(X)
            else:
                pass

        return X, y