from albumentations import *


def train_aug(p=1.0, image_size=256):
    return Compose([
        Resize(image_size, image_size),
        # GaussNoise(),
        # HorizontalFlip(0.5),
        # Rotate(limit=10),
        # Normalize(),
    ], p=p)


def valid_aug(p=1.0, image_size=256):
    return Compose([
        Resize(image_size, image_size),
        # Normalize(),
    ], p=p)


def test_aug(p=1.0, image_size=256):
    return Compose([
        Resize(image_size, image_size),
        # Normalize(),
    ], p=p)


def test_tta(p=1.0, image_size=256):
    return [
        Compose([
            Resize(image_size, image_size),
            HorizontalFlip(p=1),
        ], p=p),
        Compose([
            Resize(image_size, image_size),
        ], p=p),
    ]