import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# === 1. Шляхи до файлів ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# === 2. Перевірка файлів ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Не знайдено файл моделі: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"❌ Не знайдено файл класів: {LABELS_PATH}")

# === 3. Завантаження моделі ===
model = tf.keras.models.load_model(MODEL_PATH)

# === 4. Завантаження назв класів ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

# Сортуємо за індексами, щоб порядок збігався
CLASS_NAMES = [label for label, _ in sorted(labels_dict.items(), key=lambda x: x[1])]
print(CLASS_NAMES)
# === 5. Основна функція ===
def predict_sign(image_path):
    """
    Передбачення дорожнього знака
    :param image_path: шлях до зображення
    :return: (назва_знака, точність)
    """
    try:
        img = image.load_img(image_path, target_size=(128, 128))  # має відповідати тренуванню
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        predicted_label = CLASS_NAMES[predicted_index]
        return predicted_label, round(confidence, 2)

    except Exception as e:
        print(f"❌ Помилка під час передбачення: {e}")
        return "Помилка розпізнавання", 0.0


'''
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# === 1. Шляхи до файлів ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# === 2. Перевірка файлів ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Не знайдено файл моделі: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"❌ Не знайдено файл класів: {LABELS_PATH}")

# === 3. Завантаження моделі ===
model = tf.keras.models.load_model(MODEL_PATH)

# === 4. Завантаження назв класів ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

# ✅ labels.json у форматі {"0": "Назва", "1": "Назва", ...}
# Отже, просто формуємо список назв у порядку зростання індексів
CLASS_NAMES = [labels_dict[str(i)] for i in range(len(labels_dict))]

# === 5. Основна функція ===
def predict_sign(image_path):
    """
    Передбачення дорожнього знака
    :param image_path: шлях до зображення
    :return: (назва_знака, точність)
    """
    try:
        img = image.load_img(image_path, target_size=(128, 128))  # має відповідати тренуванню
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        predicted_label = CLASS_NAMES[predicted_index]
        return predicted_label, round(confidence, 2)

    except Exception as e:
        print(f"❌ Помилка під час передбачення: {e}")
        return "Помилка розпізнавання", 0.0


''''''
# Імена класів (підстав свої)
CLASS_NAMES = [
"Обмеження швидкості (20 км/год)", 0
"Обмеження швидкості (30 км/год)", 1
"Обмеження швидкості (50 км/год)", 2
"Обмеження швидкості (60 км/год)", 3
"Обмеження швидкості (70 км/год)", 4
"Обмеження швидкості (80 км/год)", 5
"Кінець обмеження швидкості (80 км/год)", 6
"Обмеження швидкості (100 км/год)", 7 
"Обмеження швидкості (120 км/год)", 8
"Обгін заборонено", 9
"Обгін заборонено для транспортних засобів масою понад 3,5 тонни", 10
"Перевага на перехресті", 11
"Головна дорога", 12
"Дати дорогу", 13 
"Стоп", 14
"Рух транспортних засобів заборонено", 15
"Рух транспортних засобів масою понад 3,5 тонни заборонено", 16
"В’їзд заборонено", 17
"Загальна небезпека", 18
"Небезпечний поворот ліворуч", 19
"Небезпечний поворот праворуч", 20
"Подвійний поворот", 21
"Нерівна дорога", 22
"Слизька дорога", 23
"Звуження дороги праворуч", 24
"Дорожні роботи", 25
"Світлофорне регулювання", 26
"Пішоходи", 27
"Діти на дорозі", 28
"Велосипедисти на дорозі", 29
"Обережно: лід або сніг", 30
"Дикі тварини на дорозі", 31
"Кінець усіх обмежень швидкості та обгону", 32
"Поворот ліворуч попереду", 33
"Поворот праворуч попереду", 34
"Рух тільки прямо", 35
"Рух прямо або праворуч", 36
"Рух прямо або ліворуч", 37
"Триматися праворуч", 38
"Триматися ліворуч", 39
"Рух по колу обов’язковий", 40
"Кінець заборони обгону", 41
"Кінець заборони обгону для транспортних засобів масою понад 3,5 тонни" 42

]
'''
