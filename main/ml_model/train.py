import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# === 1. Автоматичне визначення шляхів ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '../../dataset')

train_dir = os.path.join(DATASET_DIR, 'train')
test_dir = os.path.join(DATASET_DIR, 'test')

MODEL_PATH = os.path.join(BASE_DIR, 'sign_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.json')

# === 2. Налаштування параметрів ===
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 15  # Більше епох для кращого навчання

# === 3. Генератори даних з посиленою аугментацією ===
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=False,  # дорожні знаки рідко дзеркально перевертаються
    brightness_range=[0.7,1.3]
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === 4. Завантаження тестових зображень ===
def load_test_images(test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    images, filenames = [], []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, file)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            filenames.append(file)
    return np.array(images), filenames

X_test, test_filenames = load_test_images(test_dir)

print(f"🔹 Кількість зображень у train: {train_data.samples}")
print(f"🔹 Кількість зображень у val: {val_data.samples}")
print(f"🔹 Кількість зображень у test: {len(X_test)}")

# === 5. Збереження мапи класів ===
with open(LABELS_PATH, 'w', encoding='utf-8') as f:
    json.dump(train_data.class_indices, f, ensure_ascii=False, indent=2)
print(f"✅ Класи збережено у файл {LABELS_PATH}")

# === 6. Побудова моделі з BatchNormalization ===
model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Оптимізатор
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# === 7. Класовий баланс для менш чисельних класів ===
classes = np.array(list(train_data.class_indices.values()))
y_train = train_data.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# === 8. Навчання з EarlyStopping та ModelCheckpoint ===
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop]
)

print(f"✅ Тренування завершено! Модель збережено у: {MODEL_PATH}")

# === 9. Передбачення для тестових даних ===
# === 9. Передбачення для тестових даних з порогом впевненості ===
CONFIDENCE_THRESHOLD = 0.7  # поріг впевненості

if len(X_test) > 0:
    predictions = model.predict(X_test)
    predicted_classes = []
    
    # Завантажуємо мапу класів для виводу назв
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    # Перевертаємо словник, щоб отримати назви класів по індексу
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    for pred in predictions:
        max_prob = np.max(pred)
        if max_prob >= CONFIDENCE_THRESHOLD:
            predicted_classes.append(idx_to_class[np.argmax(pred)])
        else:
            predicted_classes.append("невизначено")
    
    # Виводимо результати
    for filename, pred_class in zip(test_filenames, predicted_classes):
        print(f"{filename} -> {pred_class} (впевненість: {max_prob:.2f})")
        
    print(f"✅ Отримано передбачення для {len(predicted_classes)} тестових зображень")
else:
    print("⚠️ Тестові зображення відсутні або не знайдені.")
