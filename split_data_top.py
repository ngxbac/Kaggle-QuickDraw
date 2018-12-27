import pandas as pd
import os
from tqdm import *


def main():
    root_csv = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/csv/train_simplified/"
    split_csv = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/all_data/"
    files = os.listdir(root_csv)

    for file in tqdm(files):
        # print(file)
        file_path = os.path.join(root_csv, file)
        df = pd.read_csv(file_path, usecols=["key_id", "drawing", "recognized", "countrycode"])
        nrows = df.shape[0]
        # Spend 5000 rows for validation
        n_train = nrows - 5000
        df_train = df.head(n_train)
        df_valid = df.tail(5000)

        path = os.path.join(split_csv, "train")
        os.makedirs(path, exist_ok=True)
        df_train.to_csv(os.path.join(path, file), index=False)

        path = os.path.join(split_csv, "valid")
        os.makedirs(path, exist_ok=True)
        df_valid.to_csv(os.path.join(path, file), index=False)


if __name__ == '__main__':
    main()
