import os
import argparse
import pandas as pd
import numpy as np


def parser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help='Metadata of training data',
                        default='../dataset/train/train_meta.csv')

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_argument()

    metadata_dir = os.path.dirname(args.metadata)
    df = pd.read_csv(args.metadata)

    df_mask = df[~np.isnan(df['mask'])]
    df_dis = df[~np.isnan(df['distancing'])]

    df_mask.to_csv(os.path.join(metadata_dir, 'mask_train_meta.csv'))
    df_dis.to_csv(os.path.join(metadata_dir, 'dis_train_meta.csv'))
