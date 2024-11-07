import os
from flask import Flask, render_template, request, session
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Set the secret key for session management
app.secret_key = os.urandom(24)  # Random secret key for session security

UPLOAD_FOLDER = "C:/Skin disease with SVM/static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load your trained models
feature_extractor_model = load_model('mymodel/feature_extractor_model.keras')
svm_classifier = joblib.load('mymodel/svm_classifier.joblib')
le = LabelEncoder()
le.classes_ = np.load('label_classes.npy', allow_pickle=True)
# Function to handle no-cache header
@app.after_request
def add_no_cache_header(response):
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

# Function to clear session before each request
@app.before_request
def clear_session():
    session.clear()  # Clears session on every page load
# Add the no-cache header to all responses
@app.after_request
def add_no_cache_header(response):
    response.cache_control.no_cache = True
    return response

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    prediction_text = None  # Ensure that prediction_text starts as None every time

    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            # Save the uploaded image
            image_location = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
            image_file.save(image_location)

            # Load and preprocess the image
            img = cv2.imread(image_location)
            if img is None:
                return render_template("index.html", prediction="Error: Image could not be loaded. Please upload a valid image file.")

            # Resize and preprocess the image
            try:
                img = cv2.resize(img, (224, 224))
            except cv2.error as e:
                return render_template("index.html", prediction=f"Error in resizing image: {e}")

            # Convert image to the required format and preprocess
            img = np.array([img])  # Wrap in array for batch processing
            img = preprocess_input(img)

            # Extract features and make prediction
            features = feature_extractor_model.predict(img)
            features_flattened = features.reshape(1, -1)
            predicted_class_index = svm_classifier.predict(features_flattened)[0]
            prediction_text = le.classes_[predicted_class_index]  # Set prediction_text after prediction

    return render_template("index.html", prediction=prediction_text)  # Pass prediction_text as template variable

if __name__ == "__main__":
    app.run(port=12000, debug=True)
