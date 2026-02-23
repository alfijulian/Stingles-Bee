import tensorflow as tf
from dataset import load_datasets
from model import build_model
from config import *

train_ds, val_ds = load_datasets()
model, base_model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Stage 1
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1)

# Fine tuning
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2)

model.save("models/final_model.keras")
