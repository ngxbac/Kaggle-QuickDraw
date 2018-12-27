import pandas as pd
import os
from tqdm import *


def merge_data(data_clean_out, data_recgonized, data_clean):
    files = os.listdir(data_recgonized)
    for file in tqdm(files):
        # print(file)
        file_path = os.path.join(data_recgonized, file)
        df_recognized = pd.read_csv(file_path)

        file_path = os.path.join(data_clean_out, file)
        df_clean_unrecognized = pd.read_csv(file_path)

        df_clean = pd.concat([df_recognized, df_clean_unrecognized], axis=0)

        path = os.path.join(data_clean)
        os.makedirs(path, exist_ok=True)
        df_clean.to_csv(os.path.join(path, file), index=False)


def main():
    data_clean_train_out = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_unrecognized/train_clean/"
    data_clean_valid_out = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_unrecognized/valid_clean/"

    data_train_recognized = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_recognized/train/"
    data_valid_recognized = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_recognized/valid/"

    data_train_clean = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean/train/"
    data_valid_clean = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean/valid/"

    # Merge train
    merge_data(data_clean_train_out, data_train_recognized, data_train_clean)

    # Merge valid
    merge_data(data_clean_valid_out, data_valid_recognized, data_valid_clean)


if __name__ == '__main__':
    main()