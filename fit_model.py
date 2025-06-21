import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from PIL import Image

# --- Configuración general ---
BASE_DIR = os.path.dirname(__file__)
TRAIN_CSV = os.path.join(BASE_DIR, "data", "Train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_GTSRB_Lite.keras")

IMG_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 43


# --- Cargar dataset ---
df = pd.read_csv(TRAIN_CSV)


def load_and_preprocess_image(path):
    img_path = os.path.join(BASE_DIR, "data", path)
    img = Image.open(img_path).convert("RGB")  # Carga imagen en modo RGB
    img = img.resize(
        (IMG_SIZE, IMG_SIZE), Image.BICUBIC
    )
    img = np.array(img) / 255.0
    return img


print("Cargando y preprocesando imagenes...")
X = np.array([load_and_preprocess_image(p) for p in df["Path"]])
y = to_categorical(df["ClassId"], NUM_CLASSES)

# --- Split de datos ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Generador con data augment ---
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
)
datagen.fit(X_train)

# --- Definición del modelo ---
model = Sequential(
    [
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# --- Entrenamiento ---
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[early_stop],
)

# --- Guardar modelo ---
model.save(MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")

# --- Gráficas de entrenamiento ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Pérdida")
plt.xlabel("Épocas")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Precisión")
plt.xlabel("Épocas")
plt.legend()

plt.tight_layout()
plt.show()
