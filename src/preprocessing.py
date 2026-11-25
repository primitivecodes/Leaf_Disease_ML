"""Data processing: loading, augmentation, single-image preprocessing, merging uploaded data."""
import os, shutil
import tensorflow as tf

IMG_SIZE = (160, 160)
BATCH = 32
IMG_EXT = (".jpg", ".jpeg", ".png")

def get_datasets(train_dir="data/train", test_dir="data/test", val_split=0.2, seed=42):
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="training", seed=seed,
        image_size=IMG_SIZE, batch_size=BATCH)
    val = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="validation", seed=seed,
        image_size=IMG_SIZE, batch_size=BATCH)
    test = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH, shuffle=False)
    names = train.class_names
    ac = tf.data.AUTOTUNE
    return train.prefetch(ac), val.prefetch(ac), test.prefetch(ac), names

def augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
    ])

def preprocess_bytes(data: bytes):
    """Raw image bytes -> model-ready batch (1, H, W, 3). Scaling is done inside the model."""
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return tf.expand_dims(img, 0).numpy()

def count_images(folder):
    out = {}
    if not os.path.isdir(folder):
        return out
    for c in sorted(os.listdir(folder)):
        p = os.path.join(folder, c)
        if os.path.isdir(p):
            out[c] = len([f for f in os.listdir(p) if f.lower().endswith(IMG_EXT)])
    return out

def merge_uploads(upload_dir="data/uploads", train_dir="data/train"):
    """Move uploaded, labelled images into the training set. Returns number merged."""
    n = 0
    for c, k in count_images(upload_dir).items():
        if k == 0:
            continue
        dst = os.path.join(train_dir, c)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(os.path.join(upload_dir, c)):
            if f.lower().endswith(IMG_EXT):
                shutil.move(os.path.join(upload_dir, c, f), os.path.join(dst, f))
                n += 1
    return n
