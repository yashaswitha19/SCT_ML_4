import numpy as np
import tensorflow as tf

# ==========================================================
# LOAD MODEL
# ==========================================================

model = tf.keras.models.load_model("hand_gesture_model.keras")

# ==========================================================
# CLASS NAMES
# ==========================================================

class_names = [
    "Palm",
    "L",
    "Fist",
    "Fist Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm Moved",
    "C",
    "Down"
]

# ==========================================================
# IMAGE PATH
# ==========================================================

image_path = "dataset/leapGestRecog/00/01_palm/frame_00_01_0001.png"   # Change this path

# ==========================================================
# LOAD & PREPROCESS IMAGE
# ==========================================================

try:
    image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
except Exception as e:
    print("Error loading image:", e)
    exit()

image = tf.keras.utils.img_to_array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

# ==========================================================
# PREDICTION
# ==========================================================

prediction = model.predict(image, verbose=0)

predicted_index = np.argmax(prediction)
confidence = np.max(prediction)
gesture = class_names[predicted_index]

# ==========================================================
# OUTPUT
# ==========================================================

print("=" * 40)
print(f"Predicted Gesture : {gesture}")
print(f"Confidence        : {confidence * 100:.2f}%")
print("=" * 40)