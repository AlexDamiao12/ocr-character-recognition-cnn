import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

MAT_FILE = os.path.join("matlab", "emnist-balanced.mat")
print("Carregando dados do arquivo .mat...")
emnist = scipy.io.loadmat(MAT_FILE)

#Extrair
train_data = emnist['dataset']['train'][0, 0]
test_data = emnist['dataset']['test'][0, 0]
mapping = emnist['dataset']['mapping'][0, 0]

label_map = {row[0]: row[1] for row in mapping}  

x_train = train_data['images'][0, 0]
y_train = train_data['labels'][0, 0].flatten()
x_test = test_data['images'][0, 0]
y_test = test_data['labels'][0, 0].flatten()

#Pré-processamento
x_train = x_train.reshape((-1, 28, 28), order='F')
x_test = x_test.reshape((-1, 28, 28), order='F')
x_train = np.rot90(x_train, k=3, axes=(1, 2)) / 255.0
x_test = np.rot90(x_test, k=3, axes=(1, 2)) / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

num_classes = len(np.unique(y_train))

#Modelo
model = tf.keras.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Treinamento
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop]
)

# 5. Avaliação
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Precisão nos dados de teste: {test_acc:.4f}")

# 6. Salvar modelo e mapeamento
model.save("modelo_emnist_balanced.keras")
np.save("label_map.npy", label_map)

# 7. Gráficos
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Precisão')
plt.xlabel('Época')
plt.ylabel('Precisão')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("grafico_resultado.png")
plt.show()