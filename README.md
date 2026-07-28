# ✋ Hand Gesture Recognition using CNN

A deep learning-based Hand Gesture Recognition system built using **TensorFlow/Keras**. The model classifies hand gesture images into one of **10 predefined gesture classes** using a Convolutional Neural Network (CNN).

---

## 📌 Project Overview

This project trains a CNN to recognize static hand gestures from images. It demonstrates image preprocessing, CNN model development, training, evaluation, and gesture prediction.

---

## 🚀 Features

- Hand gesture image classification
- Custom CNN architecture
- Image preprocessing and normalization
- Model evaluation with accuracy and classification report
- Predict gesture from a single image
- Easy-to-use prediction script

---

## 📂 Dataset

**Dataset:** LeapGestRecog

The dataset contains hand gesture images belonging to the following classes:

- Palm
- L
- Fist
- Fist Moved
- Thumb
- Index
- OK
- Palm Moved
- C
- Down

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## 📁 Project Structure

```
SCT_ML_4/
│
├── train.py                # Model training
├── predict.py              # Predict gesture from image
├── requirements.txt
├── README.md
├── .gitignore
└── dataset/                # Dataset (not included)
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yashaswitha19/SCT_ML_4.git
```

Move into the project directory

```bash
cd SCT_ML_4
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## 🔍 Predict a Gesture

Update the image path inside `predict.py` and run:

```bash
python predict.py
```

Example Output

```
========================================
Predicted Gesture : Palm
Confidence        : 99.98%
========================================
```

---

## 📊 Model Performance

- Validation Accuracy: **100%**
- Validation Loss: **0.000008**

*(Performance may vary depending on dataset split and training configuration.)*

---

## 📈 Future Improvements

- Real-time webcam gesture recognition
- Transfer learning with MobileNet/EfficientNet
- Support for more gesture classes
- Improved generalization on real-world images
- Deployment as a web application

---

## 🎯 Learning Outcomes

- Image preprocessing
- CNN architecture design
- Multi-class image classification
- TensorFlow/Keras workflow
- Model evaluation and inference

---

