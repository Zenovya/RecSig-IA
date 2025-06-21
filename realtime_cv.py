import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image

# --- Configuración ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_GTSRB_Lite.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

IMG_SIZE = 30
ROI_SIZE = 256
THRESHOLD = 0.92

# --- Cargar modelo y etiquetas ---
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

confidencias = deque(maxlen=3)

# --- Configuración de la cámara ---
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -- Calcular ROI centrado en la imagen ---
center_x = frame_width // 2
center_y = frame_height // 2
x1 = max(0, center_x - ROI_SIZE // 2)
y1 = max(0, center_y - ROI_SIZE // 2)
x2 = min(frame_width, center_x + ROI_SIZE // 2)
y2 = min(frame_height, center_y + ROI_SIZE // 2)

print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(roi).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img_array = np.array(img_pil) / 255.0
    roi_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(roi_input, verbose=0)
    pred_idx = int(np.argmax(pred))
    conf = float(np.max(pred))
    confidencias.append(conf)
    media_conf = sum(confidencias) / len(confidencias)
    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"Clase {pred_idx}"

    texto = (
        f"{pred_label} ({media_conf:.2f})"
        if media_conf >= THRESHOLD
        else "Sin reconocimiento confiable"
    )

    if media_conf >= THRESHOLD:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        max_width = frame.shape[1] - 100

        # Función para dividir texto en varias líneas
        def wrap_text(text, max_width):
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                (line_width, _), _ = cv2.getTextSize(
                    test_line, font, font_scale, thickness
                )
                if line_width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            return lines

        lines = wrap_text(texto, max_width)

        # Mostrar líneas centradas
        line_height = 18
        start_y = max(y1 - 10 - (len(lines) - 1) * line_height, 20)
        for i, line in enumerate(lines):
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_x = max((frame.shape[1] - text_width) // 2, 10)
            text_y = start_y + i * line_height
            cv2.putText(
                frame, line, (text_x, text_y), font, font_scale, (0, 0, 0), thickness
            )

        print(f"Predicción: {pred_label}, Confianza: {media_conf:.2f}")
    else:
        cv2.putText(
            frame,
            texto,
            (x1 - 10, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )

    bar_length = int(255 * media_conf)
    cv2.rectangle(
        frame,
        (x1, y2 + 10),
        (x1 + bar_length, y2 + 30),
        (0, 0, 255) if media_conf < THRESHOLD else (0, 255, 0),
        -1,
    )
    cv2.putText(
        frame,
        f"{media_conf:.2f}",
        (x1, y2 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255) if media_conf < THRESHOLD else (0, 255, 0),
        2,
    )

    resized_frame = cv2.resize(
        frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR
    )
    cv2.imshow("Reconocimiento de señales - Presiona 'q' para salir", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
