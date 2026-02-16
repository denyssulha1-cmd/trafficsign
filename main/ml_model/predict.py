import os
import json
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# === 1. Шляхи ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.tflite")  # Тепер TFLite
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# === 2. Завантаження класів ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

CLASS_NAMES = [labels_dict[str(i)] for i in range(len(labels_dict))]

# === 3. Завантаження моделі TFLite ===
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === 4. Основна функція передбачення ===
def predict_sign(image_path):
    try:
        # Завантаження і нормалізація
        img = Image.open(image_path).resize((128, 128))
        img_array = np.array(img).astype(np.float32) / 255.0

        # Якщо сіре зображення, зробити 3 канали
        if img_array.ndim == 2:
            img_array = np.stack([img_array]*3, axis=-1)

        # Додаємо batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Передбачення через TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0])) * 100
        predicted_label = CLASS_NAMES[predicted_index]

        return predicted_label, round(confidence, 2)

    except Exception as e:
        print(f"❌ Помилка передбачення: {e}")
        return "Помилка розпізнавання", 0.0
