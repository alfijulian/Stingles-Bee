from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionResNetV2
from config import IMAGE_SIZE, NUM_CLASSES

def build_model():
    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES)(x)

    model = keras.Model(inputs, outputs)
    return model, base_model
