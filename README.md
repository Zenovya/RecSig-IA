# 🚦 GTSRB Lite - Reconocimiento de Señales de Tránsito

Este proyecto permite entrenar, evaluar y ejecutar en tiempo real un modelo de reconocimiento de señales de tránsito utilizando el dataset GTSRB (German Traffic Sign Recognition Benchmark).

## 📦 Requisitos previos

1. Python 3.9 o superior
2. Acceso a [Repositorio del dataset GTSRB](https://www.kaggle.com/)

## 🧰 1. Preparación del entorno

### Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate     # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📁 2. Descargar y preparar los datos

### Descargar el dataset desde Kaggle

1. Accede a: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
2. Descarga el dataset completo.
3. Extrae el contenido y copia los archivos `Train.csv`, `Test.csv` y las carpeta `Train/`, `Test/` dentro de la carpeta `data/` del proyecto.

La estructura resultante debe ser:

```
project/
└── data/
    ├── Train.csv
    ├── Test.csv
    └── train/
        └── 0/
        └── 1/
        ...
        └── 42/
```

---

## 🔍 3. Verificar las imágenes

Ejecuta el siguiente script para contar y comprobar las imágenes por clase:

```bash
python img_count.py
```

---

## 🧠 4. Entrenar el modelo

1. Este paso entrena la red neuronal con los datos de `Train.csv`, y comienza a preprocesar las imagenes.
2. Visualiza el avance del entrenamiento durante cada época hasta completar el proceso.
3. Guarda el modelo en la carpeta `model/`.

```bash
python fit_model.py
```

## 🧪 5. Evaluar el modelo

Este script carga el modelo entrenado y evalúa su rendimiento usando los datos de `Test.csv`.

```bash
python evaluate_test.py
```

## 🎥 6. Ejecutar detección en tiempo real con cámara

El script `realtime_cv.py` activa la cámara y realiza predicciones en tiempo real sobre una región centrada.

```bash
python realtime_cv.py
```

> Nota: Usa la cámara 1 por defecto (`cv2.VideoCapture(1)`). Puedes cambiarlo a `cv2.VideoCapture(0)` si tu cámara está en otro puerto.

## 📂 Directorios importantes

- `data/`: contiene los datos descargados del dataset
- `model/`: directorio para guardar los modelos entrenados
- `labels.txt`: contiene los nombres de las clases (43 señales)

> ⚠️ **Nota:** Puede personalizar los nombres de los directorios según sus necesidades, siempre y cuando actualice las rutas correspondientes en los scripts para reflejar estos cambios.
