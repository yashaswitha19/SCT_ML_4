import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "dataset/leapGestRecog"

IMAGE_SIZE = 128

BATCH_SIZE = 32

MODEL_PATH = "hand_gesture_model.keras"

os.makedirs("plots", exist_ok=True)

# ==========================================================
# LOAD IMAGE PATHS
# ==========================================================

image_paths = []
labels = []

for person in sorted(os.listdir(DATASET_PATH)):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in sorted(os.listdir(person_path)):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for image in os.listdir(gesture_path):

            image_paths.append(os.path.join(gesture_path, image))

            labels.append(gesture)

# ==========================================================
# LABEL ENCODING
# ==========================================================

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

class_names = encoder.classes_

NUM_CLASSES = len(class_names)

# ==========================================================
# TRAIN / VALIDATION SPLIT
# ==========================================================

_, val_paths, _, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

# ==========================================================
# LOAD IMAGES
# ==========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image / 255.0

    label = tf.one_hot(label, depth=NUM_CLASSES)

    return image, label

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

validation_dataset = (
    validation_dataset
    .map(load_image)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel Loaded Successfully!\n")

# ==========================================================
# EVALUATE
# ==========================================================

loss, accuracy = model.evaluate(validation_dataset)

print(f"\nValidation Loss     : {loss:.6f}")
print(f"Validation Accuracy : {accuracy*100:.2f}%")

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(validation_dataset)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = np.concatenate([
    np.argmax(y.numpy(), axis=1)
    for _, y in validation_dataset
])

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("plots/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved to plots/confusion_matrix.png")