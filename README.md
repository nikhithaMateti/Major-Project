# ASL Alphabet Image Recognition
The **American Sign Language (ASL) Alphabet Image Recognition** project is an image classification system designed to recognize hand gestures representing the ASL alphabet (A–Z) along with special gestures for **space**, **delete**, and **nothing**. The goal is to bridge communication gaps between deaf and hearing communities through gesture-based recognition using computer vision and deep learning.

## 🧠 Project Overview
This project uses a **Convolutional Neural Network (CNN)**, specifically **VGG16**, for feature extraction and classification. The trained model is integrated into a **Flask web application** to enable image-based predictions in a user-friendly UI.


## 📌 Features
- Classifies 29 ASL hand signs (A-Z, space, delete, nothing)
- Utilizes transfer learning with **VGG16**
- Trains on custom image dataset from Kaggle
- Implements model evaluation and visualization via **TSNE**
- Deploys model predictions in a Flask-based web interface

---

## 📁 Dataset
**Source**: [ASL Alphabet Dataset - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)


## 🏗️ Project Architecture
User UI (Flask)
    ↓
Image Upload
    ↓
Preprocessing & Resizing (64x64 RGB)
    ↓
Prediction via VGG16-based CNN
    ↓
Output Class Label (A-Z, del, nothing, space)

**Tech Stack**
Python 3.x
TensorFlow / Keras
OpenCV
Matplotlib / Plotly / Seaborn
Flask (for deployment)
scikit-learn
TSNE for feature visualization

**Download and Extract Dataset**
!kaggle datasets download -d grassknoted/asl-alphabet
!unzip asl-alphabet.zip -d asl-alphabet


