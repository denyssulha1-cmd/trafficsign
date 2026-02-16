import tensorflow as tf

# Шлях до твоєї Keras-моделі
keras_model_path = "sign_model.h5"

# Завантажуємо модель
model = tf.keras.models.load_model(keras_model_path)

# Шлях для TFLite моделі
tflite_model_path = "sign_model.tflite"

# Конвертація у TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Опціонально) оптимізація моделі для CPU
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Генерація TFLite моделі
tflite_model = converter.convert()

# Запис у файл
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ Конвертація завершена! Файл:", tflite_model_path)
