"""CIFAR10 dataset."""

import os
import pickle

import numpy as np


import torch.utils.data
#from newnet.config import cfg



# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]


class Cifar10(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split, transformer):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(split)

        self._data_path, self._split = data_path, split

        self._inputs, self._labels = self._load_data()

    def _load_data(self):
        """Loads data into memory."""
    
        # Compute data batch names
        if self._split == "train":
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            with open(batch_path, "rb") as f:
                data = pickle.load(f, encoding="bytes")
            inputs.append(data[b"data"])
            labels += data[b"labels"]

        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)

        inputs = inputs.reshape((-1, 3, 32, 32))
        return inputs, labels

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]

        return im, label

    def __len__(self):
        return self._inputs.shape[0]

if __name__ =='__main__':
    datapath = './data/cifar10'
    cifar = Cifar10(datapath, 'train')












