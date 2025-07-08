from flask import Flask, render_template, request
import os
import numpy as np
import string
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Config
IMG_HEIGHT, IMG_WIDTH = 64, 64
labels = list(string.ascii_uppercase) + ['del', 'nothing', 'space']
pretty_labels = {chr(65+i): chr(65+i) for i in range(26)}
pretty_labels.update({'del': 'Delete', 'space': 'Space', 'nothing': 'Nothing'})

# Load model
def load_model():
    base_model = VGG16(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(29, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights('asl_vgg16_best_weights.h5')
    return model

model = load_model()

# Predict function
def predict_label(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # ✅ Proper preprocessing
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    label_index = np.argmax(preds)
    label = labels[label_index]
    pretty_label = pretty_labels.get(label, label)
    confidence = float(np.max(preds))
    return pretty_label, confidence

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            label, confidence = predict_label(filepath)
            return render_template('index.html',
                                   filename=filename,
                                   label=label,
                                   confidence=round(confidence * 100, 2))
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
