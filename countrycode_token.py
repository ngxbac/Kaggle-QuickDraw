import os
import numpy as np
import pandas as pd
from tqdm import *
from sklearn.preprocessing import LabelEncoder


def main():
    root_csv = "./data/csv/train_simplified/"
    split_train_csv = "./data/split/train_4/" #Test with version 3k first
    split_valid_csv = "./data/split/valid_4/"  # Test with version 3k first

    test_csv = "/media/ngxbac/Bac/competition/kaggle/quickdraw/data/csv/test_simplified.csv"

    split_train_token = "./data/split/train_4_token"
    os.makedirs(split_train_token, exist_ok=True)

    split_valid_token = "./data/split/valid_4_token"
    os.makedirs(split_valid_token, exist_ok=True)

    test_token_dir = "./data/split/test_4_token"
    os.makedirs(test_token_dir, exist_ok=True)

    train_codes = []
    valid_codes = []

    train_countrycode_dict = {}
    valid_countrycode_dict = {}

    files = os.listdir(root_csv)
    for file in tqdm(files):
        file_path = os.path.join(root_csv, file)
        train_csv_path = os.path.join(split_train_csv, file)
        valid_csv_path = os.path.join(split_valid_csv, file)

        df = pd.read_csv(file_path, usecols=["key_id", "countrycode"])

        train_df = pd.read_csv(train_csv_path, usecols=["key_id"])
        valid_df = pd.read_csv(valid_csv_path, usecols=["key_id"])

        train_countrycode = df.loc[df["key_id"].isin(train_df["key_id"].values.tolist())]["countrycode"].values.tolist()
        valid_countrycode = df.loc[df["key_id"].isin(valid_df["key_id"].values.tolist())]["countrycode"].values.tolist()

        train_countrycode_dict[file] = train_countrycode
        valid_countrycode_dict[file] = valid_countrycode

        train_codes += train_countrycode
        valid_codes += valid_countrycode

    train_countrycode_unique = np.unique(train_codes).tolist()
    n_unique_country_code = len(train_countrycode_unique)
    print("Number of token: {}".format(n_unique_country_code))

    label_encoder = LabelEncoder()
    label_encoder.fit(train_codes)
    for file in tqdm(files):
        saved_name = file.split(".")[0]
        # Create token for train
        token_data = label_encoder.transform(train_countrycode_dict[file])
        token_data = np.asarray(token_data)
        np.save(os.path.join(split_train_token, saved_name + ".npy"), token_data)

        # Create token for valid
        token_data = label_encoder.transform(valid_countrycode_dict[file])
        token_data = np.asarray(token_data)
        np.save(os.path.join(split_valid_token, saved_name + ".npy"), token_data)

    # Create token for test
    test_df = pd.read_csv(test_csv, usecols=["key_id", "countrycode"])
    test_codes = test_df["countrycode"].values.tolist()
    test_token = []
    for code in test_codes:
        if code in label_encoder.classes_.tolist():
            token = label_encoder.transform([code]).tolist()
            test_token += token
        else:
            test_token.append(n_unique_country_code)

    test_token = np.asarray(test_token)
    np.save(os.path.join(test_token_dir,  "test.npy"), test_token)


if __name__ == '__main__':
    main()