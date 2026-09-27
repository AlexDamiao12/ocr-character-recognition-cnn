import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAT_FILE = os.path.join(BASE_DIR, "matlab", "emnist-balanced.mat")

emnist = scipy.io.loadmat(MAT_FILE)
train_data = emnist["dataset"]["train"][0, 0]
mapping = emnist["dataset"]["mapping"][0, 0]
label_map = {int(row[0]): int(row[1]) for row in mapping}

x_train = train_data["images"][0, 0].reshape((-1, 28, 28))
y_train = train_data["labels"][0, 0].flatten().astype(np.int32)

x_train = np.transpose(x_train, axes=(0, 2, 1))  # mesmo fix do treino.py

rng = np.random.default_rng(42)
indices = rng.choice(len(x_train), size=10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, idx in zip(axes.flat, indices):
    ax.imshow(x_train[idx], cmap="gray")
    ax.set_title(chr(label_map[int(y_train[idx])]))
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "verificacao_orientacao.png"))
plt.show()
print("Confira: cada letra/número deve estar legível e bater com o título.")