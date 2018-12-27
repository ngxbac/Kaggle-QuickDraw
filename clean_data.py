import pandas as pd
import numpy as np
import os
from tqdm import *
tqdm.pandas()


from sklearn.preprocessing import LabelEncoder


CLASS_NAME=\
['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'airplane', 'alarm_clock', 'ambulance', 'angel',
 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn',
 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee',
 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel',
 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup',
 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship',
 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser',
 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses',
 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo',
 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan',
 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger',
 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck',
 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant',
 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop',
 'leaf', 'leg', 'light_bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox',
 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito',
 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean',
 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants',
 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano',
 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond',
 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'river', 'roller_coaster', 'rollerskates', 'sailboat',
 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw',
 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear',
 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle',
 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch',
 'yoga', 'zebra', 'zigzag']


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def clean_data(data_clean, data_clean_out, data_predict_path):
    lb = LabelEncoder()
    lb.fit(CLASS_NAME)

    all_cls = lb.classes_
    idx2cls = np.zeros(all_cls.shape, dtype=object)
    for i, cls in enumerate(all_cls):
        idx2cls[i] = cls

    # Load predicted train by other models
    # train_predict = np.load("./logs/clean_model_1_resnet34/dataset.predictions.data_clean_train.npy")
    data_predict = np.load(data_predict_path)
    data_predict = softmax(data_predict, axis=1)

    # all_class = os.listdir(data_2_train)
    # Load non-clean data
    dfs = []
    for cls in CLASS_NAME:
        cls = cls.replace("_", ' ')
        df = pd.read_csv(os.path.join(data_clean, cls + ".csv"))
        df["word"] = cls.split(".")[0]
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)

    pred = np.argmax(data_predict, axis=1)
    pred_cls = idx2cls[pred]

    probs = []
    for i, predict_prob in enumerate(data_predict):
        probs.append(predict_prob[pred[i]])

    dfs["predict_word"] = pred_cls
    dfs["predict_word"] = dfs["predict_word"].apply(lambda x: x.replace("_", " "))
    dfs["prob"] = probs
    print("Predicted df: ", dfs.head())

    del data_predict
    import gc
    gc.collect()

    threshold = 0.90

    def re_assign_label(row):
        if row["recognized"] == False and row["word"] != row["predict_word"] and row["prob"] > threshold:
            row["new_word"] = row["predict_word"]
        else:
            row["new_word"] = row["word"]
        return row

    recognized_df = dfs[dfs["recognized"] == True]
    unrecognized_df = dfs[dfs["recognized"] == False]
    del dfs
    gc.collect()

    clean_df = unrecognized_df.progress_apply(re_assign_label, axis=1)
    print("Number of data is corrected is: ", (clean_df["word"] != clean_df["new_word"]).sum())
    recognized_df["new_word"] = recognized_df["word"]
    clean_dfs = pd.concat([clean_df, recognized_df], axis=0)
    del clean_df, recognized_df
    gc.collect()

    all_class = clean_dfs["new_word"].unique()
    os.makedirs(data_clean_out, exist_ok=True)

    # Save csv
    for cls in tqdm(all_class):
        cls_df = clean_dfs[clean_dfs["new_word"] == cls]
        cls_df.to_csv(data_clean_out + cls + ".csv", index=False)

    del clean_dfs
    gc.collect()


def main():
    data_clean_train = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_unrecognized/train/"
    data_clean_valid = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_unrecognized/valid/"

    data_clean_train_out = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_unrecognized/train_clean/"
    data_clean_valid_out = "/media/ngxbac/Bac/competition/kaggle/competition_data/quickdraw/data/100k_clean_unrecognized/valid_clean/"

    data_train_predict = "./logs/se_resnext101_50k_2/dataset.predictions.data_clean_train.logits.npy"
    data_valid_predict = "./logs/se_resnext101_50k_2/dataset.predictions.data_clean_valid.logits.npy"

    # Clean train
    clean_data(data_clean_train, data_clean_train_out, data_train_predict)
    # Clean validation
    clean_data(data_clean_valid, data_clean_valid_out, data_valid_predict)


if __name__ == '__main__':
    main()
