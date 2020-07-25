"""CIFAR10 dataset."""

import os
import pickle
import numpy as np
import pandas as pd
import torch.utils.data

from newnet.datasets.transformers import Transforms


SIZE = 512

class TBDatasets(torch.utils.data.Dataset):

    def __init__(self, data_path, split, transformer = None):
        self._transformer = transformer
        self._data_path = data_path
        self._split = split
        self._name, self._labels = self._load_csv()
        self._labels = self._labels.astype(np.int64)

    def _load_csv(self):
        if self._split == 'train':
            path = os.path.join(self._data_path, 'df_train.csv')
            df = pd.read_csv(path)
        elif self._split == 'test':
            path = os.path.join(self._data_path, 'df_test.csv')
            df = pd.read_csv(path)

        return df['name'], df['label']

    def _transform(self,img):
 
        if self._transformer is not None:
            img = np.concatenate((img,img,img), axis = 2)
            img = self._transformer(img)
        img = np.swapaxes(img, 0,1)
        img = np.swapaxes(img, 0,2)
        return img[0:1,:,:]

    def _get_data(self, name):
        path = os.path.join(self._data_path, 'dataset', name+'.npy')
        with open(path, 'rb') as f:
            img = np.load(path).astype(np.float32)

        img = img.reshape(SIZE,SIZE,1)
        return img

    def __getitem__(self, index):
   
        name, label = self._name[index], self._labels[index]
        img = self._get_data(name)
        img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self._labels)

if __name__ =='__main__':

    import matplotlib.pyplot as plt
    datapath = "D:\\GoogleDrive\\dataset\\radiology\\tuberculosis_xray\\"
    dataset = TBDatasets(datapath,'train', Transforms(elastic_transform=True, basic=True))
    img, label = dataset[100]
    print(img.shape)
    plt.imshow(img[0,:,:])
    plt.show()
