import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from dataset import load_datasets

_, val_ds = load_datasets()
model = keras.models.load_model("models/final_model.keras")

val_labels = np.concatenate([y for x, y in val_ds], axis=0)
pred_logits = model.predict(val_ds)
pred = np.argmax(pred_logits, axis=1)

print(confusion_matrix(val_labels, pred))
print(classification_report(val_labels, pred))
