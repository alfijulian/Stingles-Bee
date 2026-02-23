import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import TRAIN_PATH, VAL_PATH, IMAGE_SIZE, BATCH_SIZE

class RandomAugmentProb(layers.Layer):
    def __init__(self, rate=0.8):
        super().__init__()
        self.rate = rate
        self.aug = keras.Sequential([
            layers.RandomRotation(0.167),
            layers.RandomZoom((-0.2, 0.2)),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(0.15),
            layers.RandomFlip("horizontal_and_vertical"),
        ])

    def call(self, x, training=True):
        if training:
            prob = tf.random.uniform([], 0, 1)
            return tf.cond(prob < self.rate, lambda: self.aug(x), lambda: x)
        return x


def load_datasets():
    train_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=IMAGE_SIZE,
        label_mode="int",
        batch_size=None,
        shuffle=True
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH,
        image_size=IMAGE_SIZE,
        label_mode="int",
        batch_size=None,
        shuffle=False
    )

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        RandomAugmentProb(rate=0.8)
    ])

    def augment(x, y):
        return augmentation(x), y

    train_ds = train_raw.shuffle(1000).map(augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_raw.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
