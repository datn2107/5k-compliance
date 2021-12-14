import os
import numpy as np
import pandas as pd
import torch
from typing import Union, List, Tuple
from torch.utils import data
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, Compose
from sklearn.model_selection import train_test_split


class ClassificationDataset(data.Dataset):
    def __init__(self, df, img_dir, img_size):
        self.df = df
        self.img_dir = img_dir
        self.transforms = Compose([
            Resize(img_size),
            Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        image = read_image(os.path.join(self.img_dir, img_name)).float()
        image = self.transforms(image)
        label = torch.Tensor(self.df.iloc[idx, 1:])

        return image, label


ARRAY_LIKE = Union[List, Tuple, np.array]


def generate_dataloader(dfs: ARRAY_LIKE, img_dirs: ARRAY_LIKE,
                        img_size: Tuple = (300, 300), batch_size: int = 8):
    states = ['train', 'val', 'test']
    dataloader = {}
    for state, df, img_dir in zip(states, dfs, img_dirs):
        if df is None:
            dataloader[state] = None
        else:
            dataset = ClassificationDataset(df, img_dir, img_size=img_size)
            dataloader[state] = data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=(state == 'train'))  # shuffle training dataset

    return dataloader


def split_dataframe(df: pd.DataFrame, dataset_size: Tuple = (0.8, 0.1, 0.1)):
    if sum(dataset_size) != 1:
        raise ValueError("Sum dataset_size must equal 1.")

    df_train, df_test = train_test_split(df, train_size=dataset_size[0])
    if dataset_size[1] == 0:
        df_val = None
    elif dataset_size[2] == 0:
        df_val = df_test
    else:
        df_val, df_test = train_test_split(df, test_size=dataset_size[2] / sum(dataset_size[1:]))

    return df_train, df_val, df_test
