import pandas as pd
import numpy as np
import os
from tqdm import *

"""
Split to a small datasets

* 1k image per class
    - train_0
    - valid_0
    
    Resnet34:
        - Batch size 128
        - Image size 64
        CV: 0.77, LB: 0.790

* 3k image per class
    - train_1
    - valid_1
    
    Resnet34:
        - Batch size 128
        - Image size 64
        CV: 0.80, LB: 0.832

* 10k image per class
    - train_2
    - valid_2

    Resnet34:
        - Batch size 128
        - Image size 64
        CV: 0.82, LB: 0.862
        
* 20k image per class
    - train_3
    - valid_3

    Resnet34:
        - Batch size 128
        - Image size 64
        CV: 0.85, LB: 0.902
        
        - With embedding
        CV: 0.859(max), LB: 0.915
        
* 30k image per class
    - train_4
    - valid_4

    Resnet34:
        - Batch size 128
        - Image size 64
        - With embedding
        CV: 0.866, LB: 0.917
"""

def main():
    root_csv = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/csv/train_simplified/"
    split_npy = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/30k/"
    # os.makedirs(split_npy)
    files = os.listdir(root_csv)

    nrows = 30000
    for file in tqdm(files):
        # print(file)
        file_path = os.path.join(root_csv, file)
        df = pd.read_csv(file_path, usecols=["key_id", "drawing", "recognized", "countrycode"], nrows=nrows)

        df_1 = df.head(int(nrows/2))
        df_1_train = df_1.head(int(nrows/2/100*90))
        # print("df_1_train ", df_1_train.shape)

        df_1_valid = df_1.tail(int(nrows/2/100*10))
        # print("df_1_valid ", df_1_valid.shape)

        df_2 = df.tail(int(nrows/2))
        df_2_train = df_2.head(int(nrows/2/100*90))
        # print("df_2_train ", df_2_train.shape)

        df_2_valid = df_2.tail(int(nrows/2/100*10))
        # print("df_2_valid ", df_2_valid.shape)

        path = os.path.join(split_npy, "data_1", "train")
        os.makedirs(path, exist_ok=True)
        df_1_train.to_csv(os.path.join(path, file), index=False)

        path = os.path.join(split_npy, "data_1", "valid")
        os.makedirs(path, exist_ok=True)
        df_1_valid.to_csv(os.path.join(path, file), index=False)

        path = os.path.join(split_npy, "data_2", "train")
        os.makedirs(path, exist_ok=True)
        df_2_train.to_csv(os.path.join(path, file), index=False)

        path = os.path.join(split_npy, "data_2", "valid")
        os.makedirs(path, exist_ok=True)
        df_2_valid.to_csv(os.path.join(path, file), index=False)


if __name__ == '__main__':
    main()
