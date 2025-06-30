
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models


DATA_PATH = 'gesture_data/00'
IMG_SIZE = 120


gesture_folders = sorted(os.listdir(DATA_PATH))
label_map = {name: idx for idx, name in enumerate(gesture_folders)}


data = []
labels = []

for gesture in gesture_folders:
    gesture_path = os.path.join(DATA_PATH, gesture)
    for img_file in tqdm(os.listdir(gesture_path), desc=f"Loading {gesture}"):
        img_path = os.path.join(gesture_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(label_map[gesture])


data = np.array(data, dtype="float32") / 255.0
data = np.expand_dims(data, -1)
labels = np.array(labels)
labels = to_categorical(labels, num_classes=len(label_map))

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)


model = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1
)


loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=gesture_folders))


print("📌 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
