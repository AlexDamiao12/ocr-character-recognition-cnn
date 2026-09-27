import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAT_FILE = os.path.join(BASE_DIR, "matlab", "emnist-balanced.mat")
MODEL_FILE = os.path.join(BASE_DIR, "modelo_emnist_balanced.keras")
LABEL_MAP_FILE = os.path.join(BASE_DIR, "label_map.json")

if not os.path.isfile(MAT_FILE):
    raise FileNotFoundError(f"Arquivo de dados não encontrado: {MAT_FILE}")

print("Carregando dados do arquivo .mat...")
emnist = scipy.io.loadmat(MAT_FILE)

train_data = emnist["dataset"]["train"][0, 0]
test_data = emnist["dataset"]["test"][0, 0]
mapping = emnist["dataset"]["mapping"][0, 0]

label_map = {int(row[0]): int(row[1]) for row in mapping}

x_train = train_data["images"][0, 0].reshape((-1, 28, 28))
y_train = train_data["labels"][0, 0].flatten().astype(np.int32)
x_test = test_data["images"][0, 0].reshape((-1, 28, 28))
y_test = test_data["labels"][0, 0].flatten().astype(np.int32)

# preprocessing
x_train = np.rot90(x_train, k=3, axes=(1, 2))
x_test = np.rot90(x_test, k=3, axes=(1, 2))

x_train = np.ascontiguousarray(x_train[..., np.newaxis], dtype=np.float32) / 255.0
x_test = np.ascontiguousarray(x_test[..., np.newaxis], dtype=np.float32) / 255.0

# data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.08, 0.08),
])

num_classes = len(np.unique(y_train))

# better CNN model
inputs = layers.Input(shape=(28, 28, 1))

x = data_augmentation(inputs)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
history = model.fit(
    x_train,
    y_train,
    epochs=40,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Precisão nos dados de teste: {test_acc:.4f}")

model.save(MODEL_FILE)
with open(LABEL_MAP_FILE, "w", encoding="utf-8") as label_file:
    json.dump({str(key): int(value) for key, value in label_map.items()}, label_file)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Treino")
plt.plot(history.history["val_accuracy"], label="Validação")
plt.title("Precisão")
plt.xlabel("Época")
plt.ylabel("Precisão")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Treino")
plt.plot(history.history["val_loss"], label="Validação")
plt.title("Perda")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "grafico_resultado.png"))
plt.show()