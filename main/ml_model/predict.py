import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

# === 1. Шляхи ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.onnx")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# === 2. Завантаження класів ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

CLASS_NAMES = [labels_dict[str(i)] for i in range(len(labels_dict))]

# === 3. Завантаження моделі ===
session = ort.InferenceSession(MODEL_PATH)

# === 4. Основна функція передбачення ===
def predict_sign(image_path):
    try:
        # Завантаження і нормалізація
        img = Image.open(image_path).resize((128, 128))
        img_array = np.array(img).astype(np.float32) / 255.0
        if img_array.ndim == 2:  # сіре зображення
            img_array = np.stack([img_array]*3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        # Підготовка до ONNX
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_array})

        predicted_index = np.argmax(outputs[0][0])
        confidence = float(np.max(outputs[0][0])) * 100
        predicted_label = CLASS_NAMES[predicted_index]

        return predicted_label, round(confidence, 2)

    except Exception as e:
        print(f"❌ Помилка передбачення: {e}")
        return "Помилка розпізнавання", 0.0
