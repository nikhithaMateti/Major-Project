# main.py

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import cv2
import numpy as np
import pandas as pd
import string
import random
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
class CFG:
    batch_size = 64
    img_height = 64
    img_width = 64
    epochs = 10
    num_classes = 29
    img_channels = 3

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# --- Labels ---
TRAIN_PATH = "asl-alphabet/asl_alphabet_train/asl_alphabet_train"
labels = list(string.ascii_uppercase) + ["del", "nothing", "space"]

# --- Data Loading ---
def create_metadata():
    image_paths = []
    image_labels = []
    for label in labels:
        files = glob.glob(os.path.join(TRAIN_PATH, label, "*"))
        image_paths.extend(files)
        image_labels.extend([label] * len(files))
    return pd.DataFrame({"image_path": image_paths, "label": image_labels})

# --- Split Data ---
def split_data(metadata):
    X_train, X_test, y_train, y_test = train_test_split(
        metadata["image_path"], metadata["label"], test_size=0.15, stratify=metadata["label"], random_state=2023
    )
    data_train = pd.DataFrame({"image_path": X_train, "label": y_train})
    X_train, X_val, y_train, y_val = train_test_split(
        data_train["image_path"], data_train["label"], test_size=0.15/0.85, stratify=data_train["label"], random_state=2023
    )
    return (
        pd.DataFrame({"image_path": X_train, "label": y_train}),
        pd.DataFrame({"image_path": X_val, "label": y_val}),
        pd.DataFrame({"image_path": X_test, "label": y_test}),
    )

# --- Data Generators ---
def create_generators(data_train, data_val, data_test):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(data_train, x_col='image_path', y_col='label',
                                            target_size=(CFG.img_height, CFG.img_width),
                                            batch_size=CFG.batch_size, class_mode='categorical')

    val_gen = datagen.flow_from_dataframe(data_val, x_col='image_path', y_col='label',
                                          target_size=(CFG.img_height, CFG.img_width),
                                          batch_size=CFG.batch_size, class_mode='categorical')

    test_gen = datagen.flow_from_dataframe(data_test, x_col='image_path', y_col='label',
                                           target_size=(CFG.img_height, CFG.img_width),
                                           batch_size=CFG.batch_size, class_mode='categorical', shuffle=False)
    return train_gen, val_gen, test_gen

# --- Model ---
def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(CFG.img_height, CFG.img_width, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(CFG.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# --- Train ---
def train_model(model, train_gen, val_gen):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint("asl_vgg16_best_weights.h5", save_best_only=True, monitor='val_loss')
    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=CFG.epochs, steps_per_epoch=train_gen.samples // CFG.batch_size,
                        validation_steps=val_gen.samples // CFG.batch_size,
                        callbacks=[checkpoint])
    return history

# --- Evaluate ---
def evaluate_model(model, test_gen):
    scores = model.evaluate(test_gen)
    print(f"Test Accuracy: {scores[1]*100:.2f}%")

# --- Predict Image ---
def predict_image(model, img_path):
    img = load_img(img_path, target_size=(CFG.img_height, CFG.img_width))
    img_array = img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    pred_label = labels[np.argmax(predictions)]
    print(f"Prediction: {pred_label}")
    return pred_label

# --- Main Pipeline ---
def main():
    seed_everything(2023)
    print("Loading metadata...")
    metadata = create_metadata()
    print("Splitting data...")
    data_train, data_val, data_test = split_data(metadata)
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_generators(data_train, data_val, data_test)
    print("Building model...")
    model = build_model()
    print("Training model...")
    train_model(model, train_gen, val_gen)
    print("Loading best model and evaluating...")
    best_model = load_model("asl_vgg16_best_weights.h5")
    evaluate_model(best_model, test_gen)

    # Test individual predictions
    test_images = [
        "asl-alphabet/asl_alphabet_test/asl_alphabet_test/A_test.jpg",
        "asl-alphabet/asl_alphabet_test/asl_alphabet_test/N_test.jpg",
        "asl-alphabet/asl_alphabet_test/asl_alphabet_test/H_test.jpg",
        "asl-alphabet/asl_alphabet_test/asl_alphabet_test/P_test.jpg",
        "asl-alphabet/asl_alphabet_test/asl_alphabet_test/nothing_test.jpg",
    ]
    for path in test_images:
        predict_image(best_model, path)

if __name__ == "__main__":
    main()
