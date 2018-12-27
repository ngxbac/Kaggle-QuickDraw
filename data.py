import numpy as np
import pandas as pd
import os
import collections
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from augmentation import train_aug, valid_aug, test_aug, test_tta

from catalyst.utils.parse import parse_in_csvs
from catalyst.utils.factory import UtilsFactory
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose
from catalyst.data.augmentor import Augmentor
from catalyst.data.sampler import BalanceClassSampler
from catalyst.dl.datasource import AbstractDataSource

from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
from keras.preprocessing.sequence import pad_sequences

# ---- Augmentations ----

IMG_SIZE = 64


# ---- Dataset ----
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


def drawing_to_image(drawing, H, W):
    # pdb.set_trace()
    point=[]
    time =[]
    # print(len(drawing))
    for t,(x,y) in enumerate(drawing):
        point.append(np.array((x,y),np.float32).T)
        time.append(np.full(len(x),t))

    point = np.concatenate(point).astype(np.float32)
    time  = np.concatenate(time ).astype(np.int32)

    #--------
    image  = np.full((H,W,3),0,np.uint8)
    x_max = point[:,0].max()
    x_min = point[:,0].min()
    y_max = point[:,1].max()
    y_min = point[:,1].min()
    w = x_max-x_min
    h = y_max-y_min
    #print(w,h)

    s = max(w,h)
    norm_point = (point-[x_min,y_min])/s
    norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)


    #--------
    T = time.max()+1
    for t in range(T):
        p = norm_point[time==t]
        x,y = p.T
        image[y,x]=255
        N = len(p)
        for i in range(N-1):
            x0,y0 = p[i]
            x1,y1 = p[i+1]
            cv2.line(image,(x0,y0),(x1,y1),(255,255,255),1,cv2.LINE_AA)

    return image


def stroke_data(raw_strokes, STROKE_COUNT=50):
    """preprocess the string and make
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes) # string->list
    # unwrap the list
    in_strokes = [(xi,yi,i)
     for i,(x,y) in enumerate(stroke_vec)
     for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1),
                         maxlen=STROKE_COUNT,
                         padding='post').swapaxes(0, 1)


class QuickDrawDataset(Dataset):
    def __init__(self, root_split, root_token, transform=None, mode="train", test_csv=None, test_token=None):

        self.mode = mode
        self.transform = transform

        self.drawings = []
        self.tokens = []
        self.labels = []

        lb = LabelEncoder()
        lb.fit(CLASS_NAME)

        # Make drawing data
        if mode == "train":
            for cls in CLASS_NAME:
                label = lb.transform([cls])
                cls = cls.replace("_", ' ')
                df = pd.read_csv(os.path.join(root_split, cls + ".csv"))
                drawing = df["drawing"].values.tolist()
                self.drawings += drawing
                self.labels += label.tolist() * len(drawing)
        else:
            df = pd.read_csv(test_csv)
            drawing = df["drawing"].values.tolist()
            self.drawings += drawing
            self.labels = [-1] * len(self.drawings)

        if mode == "train":
            # Make token data
            for cls in CLASS_NAME:
                cls = cls.replace("_", ' ')
                token = np.load(os.path.join(root_token, cls + ".npy"))
                self.tokens += token.tolist()
        else:
            token = np.load(test_token)
            self.tokens += token.tolist()

    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, idx):

        # Get token
        token = self.tokens[idx]

        # Get drawing
        drawing = self.drawings[idx]
        stroke = stroke_data(drawing).astype(np.float32)
        # stroke = np.expand_dims(stroke, axis=0)
        stroke = np.transpose(stroke, (1, 0)).astype(np.float32)

        drawing = eval(drawing)
        label = self.labels[idx]

        # Plot the image
        img, img_gray = drawing_to_image(drawing, IMG_SIZE, IMG_SIZE)

        if self.transform:
            img = self.transform(image=img)["image"]
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            # img = np.expand_dims(img, 0).astype(np.float32)

            img_gray = self.transform(image=img_gray)["image"]
            img_gray = np.expand_dims(img_gray, 0).astype(np.float32)

        return {
            "image": img,
            "image_gray": img_gray,
            "stroke": stroke,
            "token": token,
            "targets": label
        }


class QuickDrawDatasetImgOnly(Dataset):
    def __init__(self, root_split, image_size=256, transform=None, mode="train", test_csv=None):

        self.mode = mode
        self.transform = transform
        self.image_size = image_size

        self.drawings = []
        self.labels = []

        lb = LabelEncoder()
        lb.fit(CLASS_NAME)

        # Make drawing data
        if mode == "train":
            for cls in CLASS_NAME:
                label = lb.transform([cls])
                cls = cls.replace("_", ' ')
                df = pd.read_csv(os.path.join(root_split, cls + ".csv"))
                drawing = df["drawing"].values.tolist()
                self.drawings += drawing
                self.labels += label.tolist() * len(drawing)
        else:
            df = pd.read_csv(test_csv)
            drawing = df["drawing"].values.tolist()
            self.drawings += drawing
            self.labels = [-1] * len(self.drawings)

    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, idx):
        # Get drawing
        drawing = self.drawings[idx]

        drawing = eval(drawing)
        label = self.labels[idx]

        # Plot the image
        img = drawing_to_image(drawing, self.image_size, self.image_size)

        if self.transform:
            img = self.transform(image=img)["image"]
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        return {
            "image": img,
            "targets": label
        }


class DataSource(AbstractDataSource):

    @staticmethod
    def prepare_transforms(*, mode, stage=None, use_tta=False, **kwargs):
        image_size = kwargs.get("image_size", 256)
        # print(image_size)
        if mode == "train":
            if stage in ["debug", "stage1"]:
                return train_aug(image_size=image_size)
            elif stage == "stage2":
                return train_aug(image_size=image_size)
            else:
                return train_aug(image_size=image_size)
        elif mode == "valid":
            return valid_aug(image_size=image_size)
        elif mode == "infer":
            if use_tta:
                return test_tta(image_size=image_size)
            else:
                return test_aug(image_size=image_size)

    @staticmethod
    def prepare_loaders(*, args, stage=None, **kwargs):
        print("ARGS", args.image_size)
        loaders = collections.OrderedDict()
        try:
            use_tta = args.use_tta
        except:
            use_tta = False

        # root_csv = data_params.get("root_csv", None)
        train_split = kwargs.get("train_split", None)
        valid_split = kwargs.get("valid_split", None)
        infer_csv = kwargs.get("infer_csv", None)

        train_token = kwargs.get("train_token", None)
        valid_token = kwargs.get("valid_token", None)
        infer_token = kwargs.get("infer_token", None)

        data_clean_train = kwargs.get("data_clean_train", None)
        data_clean_valid = kwargs.get("data_clean_valid", None)

        blending_data = kwargs.get("blending_data", None)

        if train_split is not None:
            train_dataset = QuickDrawDatasetImgOnly(
                train_split,
                image_size=args.image_size,
                transform=DataSource.prepare_transforms(
                    mode="train",
                    stage=stage,
                    image_size=args.image_size,
                ),
                mode="train"
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=True
            )
            loaders["train"] = train_loader
            print("Number of sample in training dataset: ", len(train_loader))

        if valid_split is not None:
            valid_dataset = QuickDrawDatasetImgOnly(
                valid_split,
                image_size=args.image_size,
                transform=DataSource.prepare_transforms(
                    mode="valid",
                    stage=stage,
                    image_size=args.image_size,
                ),
                mode="train"
            )
            valid_loader = DataLoader(
                dataset=valid_dataset,
                num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=True
            )
            loaders["valid"] = valid_loader
            print("Number of sample in valid dataset: ", len(valid_loader))

        if infer_csv is not None:
            transforms = DataSource.prepare_transforms(
                mode="infer",
                stage=stage,
                image_size=args.image_size,
                use_tta=use_tta
            )
            print(transforms)
            for i, transform in enumerate(transforms):
                infer_dataset = QuickDrawDatasetImgOnly(
                    root_split=None,
                    mode="infer",
                    image_size=args.image_size,
                    transform=transform,
                    test_csv=infer_csv
                )
                infer_loader = DataLoader(
                    dataset=infer_dataset,
                    num_workers=args.workers,
                    batch_size=args.batch_size,
                    shuffle=False
                )
                loaders[f"infer_{i}"] = infer_loader
                print("Number of sample in infer dataset: ", len(infer_loader))

        if blending_data is not None:
            transforms = DataSource.prepare_transforms(
                mode="infer",
                stage=stage,
                image_size=args.image_size,
                use_tta=use_tta
            )
            print(transforms)
            for i, transform in enumerate(transforms):
                blending_dataset = QuickDrawDatasetImgOnly(
                    blending_data,
                    image_size=args.image_size,
                    transform=transform,
                    mode="train"
                )
                valid_loader = DataLoader(
                    dataset=blending_dataset,
                    num_workers=args.workers,
                    batch_size=args.batch_size,
                    shuffle=False
                )
                loaders[f"blending_{i}"] = valid_loader
                print("Number of sample in blending dataset: ", len(valid_loader))

        return loaders
