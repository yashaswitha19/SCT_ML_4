import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Dense,
    Flatten
)

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "dataset/leapGestRecog"

IMAGE_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 25

MODEL_NAME = "hand_gesture_model.keras"

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

            image_path = os.path.join(gesture_path, image)

            image_paths.append(image_path)

            labels.append(gesture)

print(f"Total Images : {len(image_paths)}")

# ==========================================================
# LABEL ENCODING
# ==========================================================

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

class_names = encoder.classes_

NUM_CLASSES = len(class_names)

print("\nClasses")

for i, name in enumerate(class_names):
    print(i, ":", name)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(

    image_paths,
    labels,

    test_size=0.20,

    stratify=labels,

    random_state=42

)

print("\nTraining Images :", len(train_paths))
print("Validation Images :", len(val_paths))

# ==========================================================
# IMAGE LOADING FUNCTION
# ==========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image / 255.0

    label = tf.one_hot(label, depth=NUM_CLASSES)

    return image, label
# ==========================================================
# TENSORFLOW WRAPPER
# ==========================================================

def tf_loader(path, label):

    image, label = tf.numpy_function(
        load_image,
        [path, label],
        [tf.float32, tf.float32]
    )

    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    label.set_shape((NUM_CLASSES,))

    return image, label

# ==========================================================
# CREATE TF DATASETS
# ==========================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

train_dataset = (
    train_dataset
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
# ==========================================================
# BUILD CNN MODEL
# ==========================================================

model = Sequential([

    Conv2D(32, (3,3), activation="relu",
           input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512, activation="relu"),
    Dropout(0.5),

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")

])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_NAME,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

]

# ==========================================================
# TRAIN MODEL
# ==========================================================

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=callbacks

)

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(MODEL_NAME)

print("\nModel saved successfully!")

# ==========================================================
# EVALUATE MODEL
# ==========================================================

loss, accuracy = model.evaluate(validation_dataset)

print(f"\nValidation Loss : {loss:.4f}")
print(f"Validation Accuracy : {accuracy*100:.2f}%")

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(validation_dataset)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = np.concatenate(
    [np.argmax(y.numpy(), axis=1) for x, y in validation_dataset]
)

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

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

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

plt.close()

# ==========================================================
# ACCURACY PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("plots/accuracy.png")

plt.close()

# ==========================================================
# LOSS PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid(True)

plt.savefig("plots/loss.png")

plt.close()

print("\n==========================================")
print("Training Completed Successfully!")
print("==========================================")

print(f"\nModel Saved : {MODEL_NAME}")

print("Plots Saved in 'plots/' folder")