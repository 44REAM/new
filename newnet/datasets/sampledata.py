# pylint: disable=redefined-outer-name

import numpy as np
from torch.utils.data import Dataset

class SampleDataset2D(Dataset):

    def __init__(self, transforms = None, n_sample=100, channels=3, width=224, height=244):
        np.random.seed(52)
        self.data = np.random.randn(
            n_sample, channels, height, width).astype(np.float32)
        self.labels = np.array([np.random.randint(0, 2)
                                for x in range(n_sample)]).astype(np.int64)
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


    

