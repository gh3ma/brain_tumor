
# 🧠 Brain Tumor Detection using CNN

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to detect brain tumors from MRI images.

The model classifies MRI scans into two categories:

* Tumor
* No Tumor

A web application was developed using **Streamlit** to allow users to upload an MRI image and receive a prediction with confidence percentage.

---

## 📂 Project Structure

```
project/
│
├── final.py                      # Model training script
├── app.py                        # Streamlit web application
├── brain_tumor_cnn_model.keras   # Trained model
├── requirements.txt              # Required Python libraries
└── README.md                     # Project documentation
```

---

## 🗂 Dataset

The dataset was collected from Kaggle and contains MRI brain images divided into:

* `yes` → Brain tumor images
* `no` → Normal brain images

The dataset was split into:

* 80% Training
* 20% Validation

All images were resized to **150 × 150 pixels**.

---

## 🧠 Model Architecture

The CNN model consists of:

* Input Layer (150×150×3)
* Rescaling Layer (Normalization)
* 3 Convolutional Layers (ReLU activation)
* MaxPooling Layers
* Flatten Layer
* Dense Layer (128 neurons)
* Dropout Layer (to reduce overfitting)
* Output Layer (Sigmoid activation)

### Training Details:

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Epochs: 10
* Accuracy achieved: ~98–100%

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the web application

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

---

## 🖥 Application Features

* Upload MRI image
* Automatic tumor detection
* Confidence percentage display
* User-friendly interface

---

## 📊 Results

The model achieved:

* Training Accuracy ≈ 99%
* Validation Accuracy ≈ 98–100%

The loss decreased steadily, indicating good learning performance.

---

## 🔮 Future Improvements

* Use larger and more diverse datasets
* Apply transfer learning models (e.g., MobileNet, EfficientNet)
* Deploy the application online
* Add Grad-CAM visualization for better explainability

---

## 👨‍💻 Author

Brain Tumor Detection Project
Developed using TensorFlow, Keras, and Streamlit.
