import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# --- Configuración ---
BASE_DIR = os.path.dirname(__file__)
TEST_CSV = os.path.join(BASE_DIR, "data", "Test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_GTSRB_Lite.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
TEST_IMG_DIR = os.path.join(BASE_DIR, "data")

# --- Parámetros ---
IMG_SIZE = 30
NUM_CLASSES = 43


# --- Cargar modelo y labels ---
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_names = [line.strip() for line in f.readlines()]

# --- Cargar datos de test ---
test_df = pd.read_csv(TEST_CSV)

X_test = []
y_true = []

print("Cargando y preprocesando imagenes de test...")
for _, row in test_df.iterrows():
    img_path = os.path.join(TEST_IMG_DIR, row["Path"])
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img = np.array(img) / 255.0
    X_test.append(img)
    y_true.append(row["ClassId"])

X_test = np.array(X_test)
y_true = np.array(y_true)

y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- Reporte de clasificación ---
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=label_names))

# --- Matriz de confusión ---
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(20, 20))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=label_names,
#     yticklabels=label_names,
# )
# plt.xlabel("Predicción")
# plt.ylabel("Verdadero")
# plt.title("Matriz de Confusión sobre el Test Set")
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
